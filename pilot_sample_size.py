"""Sample-size pilot.

For the same set of HumanEval prompts, run self-SD (E=8) at full precision (fp16) and 4bit (nf4),
collect per-sample acceptance, and from the std of the paired Delta-alpha = alpha_fp16 - alpha_4bit
compute the minimum n that yields a 95% CI width < 0.05.

Side effect: also checks whether the full-precision 7B (~14GB) OOMs (B1/B2 pre-check).
"""
import gc, json, math, os, statistics, time
import torch

from arguments import Arguments
from generate import load_model_and_tokenizer
from data import get_data
from self_speculation.generator_base import GenerationConfig, HuggingfaceLlamaGenerator
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy

MODEL = os.environ.get("QUANTSKIP_MODEL", os.path.expanduser("~/models/layerskip-llama2-7B"))
DATASET = "human_eval"
N = 100
EXIT_LAYER = 8
NUM_SPEC = 6
SEED = 42
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LayerSkip", "logs", "pilot")
os.makedirs(OUT, exist_ok=True)


def gen_config():
    return GenerationConfig(
        max_steps=512,
        exit_layer=EXIT_LAYER,
        num_speculations=NUM_SPEC,
        generation_strategy="self_speculative",
        sample=False,
        stop_words=[],
        stop_token_ids=[],
    )


def run_precision(label, quantize, examples):
    print(f"\n=== [{label}] loading (quantize={quantize}) ===", flush=True)
    torch.cuda.reset_peak_memory_stats()
    args = Arguments(model=MODEL, quantize=quantize, seed=SEED)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(args)
    print(f"[{label}] loaded in {time.time()-t0:.0f}s | VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)
    gen = HuggingfaceLlamaGenerator(tokenizer=tok, model=model, generation_strategy=SelfSpeculativeGenerationStrategy())
    accs = []
    t1 = time.time()
    for i, ex in enumerate(examples):
        r = gen.generate(prompt=ex.input, generation_config=gen_config())
        a = r.generation_strategy_result.acceptance_rate
        accs.append(a)
        if (i + 1) % 10 == 0:
            valid = [x for x in accs if x is not None]
            m = statistics.mean(valid) if valid else float("nan")
            print(f"[{label}] {i+1}/{len(examples)} | mean acc so far {m:.3f} | {time.time()-t1:.0f}s", flush=True)
    print(f"[{label}] done | VRAM peak {torch.cuda.max_memory_allocated()/1e9:.2f}GB | {time.time()-t1:.0f}s", flush=True)
    del model, gen
    gc.collect(); torch.cuda.empty_cache()
    return accs


def main():
    examples = get_data(random_shuffle=True, num_samples=N, dataset=DATASET,
                        n_shot=0, seed=SEED, data_path=None, template=None)
    print(f"loaded {len(examples)} {DATASET} prompts", flush=True)

    # full precision first (also the OOM check), then 4bit
    acc_fp = run_precision("fp16", None, examples)
    acc_q = run_precision("nf4", "nf4", examples)

    # paired Delta-alpha (only samples valid in both)
    deltas, a_fp, a_q = [], [], []
    for x, y in zip(acc_fp, acc_q):
        if x is not None and y is not None:
            deltas.append(x - y); a_fp.append(x); a_q.append(y)

    n_valid = len(deltas)
    sd_delta = statistics.stdev(deltas) if n_valid > 1 else float("nan")
    mean_delta = statistics.mean(deltas) if deltas else float("nan")
    # CI total width < 0.05 -> half-width 0.025, z=1.96
    z, halfwidth = 1.96, 0.025
    n_req = math.ceil((z * sd_delta / halfwidth) ** 2) if sd_delta == sd_delta else None

    summary = {
        "dataset": DATASET, "exit_layer": EXIT_LAYER, "num_speculations": NUM_SPEC,
        "n_pilot": N, "n_valid_pairs": n_valid,
        "mean_acc_fp16": statistics.mean(a_fp) if a_fp else None,
        "mean_acc_nf4": statistics.mean(a_q) if a_q else None,
        "mean_delta_alpha": mean_delta,
        "std_delta_alpha": sd_delta,
        "required_n_for_CI_width_0.05": n_req,
        "acc_fp16_per_sample": acc_fp,
        "acc_nf4_per_sample": acc_q,
    }
    with open(os.path.join(OUT, "pilot_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n==================== PILOT SUMMARY ====================", flush=True)
    print(f"valid pairs           : {n_valid}/{N}", flush=True)
    print(f"mean acc  fp16 / nf4  : {summary['mean_acc_fp16']:.4f} / {summary['mean_acc_nf4']:.4f}", flush=True)
    print(f"mean delta (fp16-nf4) : {mean_delta:+.4f}", flush=True)
    print(f"std  delta (per-sample): {sd_delta:.4f}", flush=True)
    print(f"required n (CI<0.05)  : {n_req}", flush=True)
    print(f"saved -> {os.path.join(OUT, 'pilot_results.json')}", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    main()
