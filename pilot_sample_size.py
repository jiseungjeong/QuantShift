"""QuantShift 샘플 수 파일럿 (§0-6, §4).

같은 HumanEval 프롬프트 집합에 대해 self-SD(E=8)를 full precision(fp16)과 4bit(nf4)로 각각 돌려
per-sample acceptance를 모으고, 페어 Δα = α_fp16 − α_4bit 의 표준편차로 CI폭<0.05 만족 최소 n을 계산한다.

실행:
  cd LayerSkip && conda run -n NLP env PYTHONNOUSERSITE=1 PYTHONPATH=$PWD \
      python ../pilot_sample_size.py
부수효과: full precision 7B(~14GB) OOM 여부도 확인된다(B1/B2 사전점검).
"""
import gc, json, math, os, statistics, time
import torch

from arguments import Arguments
from generate import load_model_and_tokenizer
from data import get_data
from self_speculation.generator_base import GenerationConfig, HuggingfaceLlamaGenerator
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy

MODEL = os.path.expanduser("~/models/layerskip-llama2-7B")
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

    # full precision first (큰 모델 → OOM 확인 겸), 그다음 4bit
    acc_fp = run_precision("fp16", None, examples)
    acc_q = run_precision("nf4", "nf4", examples)

    # paired Δα (둘 다 valid인 샘플만)
    deltas, a_fp, a_q = [], [], []
    for x, y in zip(acc_fp, acc_q):
        if x is not None and y is not None:
            deltas.append(x - y); a_fp.append(x); a_q.append(y)

    n_valid = len(deltas)
    sd_delta = statistics.stdev(deltas) if n_valid > 1 else float("nan")
    mean_delta = statistics.mean(deltas) if deltas else float("nan")
    # CI 총폭 < 0.05 → 반폭 0.025, z=1.96
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
    print(f"mean Δα (fp16 − nf4)  : {mean_delta:+.4f}", flush=True)
    print(f"std  Δα (per-sample)  : {sd_delta:.4f}", flush=True)
    print(f"required n (CI<0.05)  : {n_req}", flush=True)
    print(f"saved -> {os.path.join(OUT, 'pilot_results.json')}", flush=True)
    print("=======================================================", flush=True)


if __name__ == "__main__":
    main()
