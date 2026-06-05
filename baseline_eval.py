"""QuantShift §4 baselines B1-B6: speed (wall-clock) + quality + memory, with latency breakdown.

For ONE precision (fp16/nf4/int8) this runs both strategies x both datasets:
  - strategy: autoregressive (B1/B3/B5) and self_speculative E=8,gamma=6 (B2/B4/B6)
  - dataset:  HumanEval -> pass@1 (exec + timeout) ;  CNN/DM -> ROUGE-2
  - records: tokens/s, time/token, VRAM peak, model load time, acceptance (self-SD only, also from shadow)
  - latency breakdown: torch.profiler on a few self-SD gens -> per-op CUDA time (bucketed later)

Quality notes:
  - HumanEval pass@1: LayerSkip data.py drops test cases, so we load `openai_humaneval` directly
    (prompt/test/entry_point), generate the completion, then exec(prompt+completion+test) under a timeout.
  - CNN/DM ROUGE-2 via torchmetrics, vs reference highlights.

Run (one precision):
  cd LayerSkip && conda run -n NLP env PYTHONNOUSERSITE=1 PYTHONPATH=$PWD \
    python ../baseline_eval.py --precision fp16 --num 100 --max-steps 128
"""
import argparse, json, os, signal, statistics, time
import torch

from arguments import Arguments
from generate import load_model_and_tokenizer
from data import get_data
from self_speculation.generator_base import GenerationConfig, HuggingfaceLlamaGenerator
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy

MODEL = os.path.expanduser("~/models/layerskip-llama2-7B")
SEED = 42
EXIT_LAYER = 8
NUM_SPEC = 6
DISK_GB = None  # filled at runtime


# ---------------- HumanEval pass@1 (exec + timeout) ----------------
class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


_STOP_SEQS = ["\ndef ", "\nclass ", "\nif __name__", "\nprint(", "\n@", "\n```", "\nassert ", "\n#"]


def truncate_completion(text):
    """Standard HumanEval: cut the generated body at the first top-level stop sequence
    so trailing garbage (new defs, prints, fences) does not break exec."""
    cut = len(text)
    for s in _STOP_SEQS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return text[:cut]


def fix_indent(completion):
    """SentencePiece decode of generated-only tokens drops one leading space, so the function body
    starts at 3-space indent instead of 4 -> IndentationError. Restore the missing space on the
    first body line (HumanEval bodies are 4-space indented)."""
    lines = completion.split("\n")
    for i, ln in enumerate(lines):
        if ln.strip() == "":
            continue
        n = len(ln) - len(ln.lstrip(" "))
        if 0 < n < 4:                       # under-indented first code line -> restore
            lines[i] = " " * (4 - n) + ln
        break
    return "\n".join(lines)


def run_passk_problem(prompt, completion, test, entry_point, timeout=8):
    """Return True if prompt+completion passes the unit test within `timeout` seconds."""
    completion = truncate_completion(fix_indent(completion))
    program = prompt + completion + "\n" + test + f"\ncheck({entry_point})\n"
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout)
    try:
        env = {}
        exec(program, env)
        return True
    except Exception:
        return False
    finally:
        signal.alarm(0)


def make_generator(model, tok, strategy):
    strat = AutoRegressiveGenerationStrategy() if strategy == "ar" else SelfSpeculativeGenerationStrategy()
    return HuggingfaceLlamaGenerator(tokenizer=tok, model=model, generation_strategy=strat)


def gen_config(strategy, max_steps):
    return GenerationConfig(
        max_steps=max_steps,
        exit_layer=EXIT_LAYER if strategy == "selfsd" else -1,
        num_speculations=NUM_SPEC if strategy == "selfsd" else -1,
        generation_strategy="self_speculative" if strategy == "selfsd" else "autoregressive",
        sample=False, stop_words=[], stop_token_ids=[],
    )


def eval_humaneval(model, tok, strategy, num, max_steps):
    from datasets import load_dataset
    ds = list(load_dataset("openai_humaneval", split="test"))[:num]
    gen = make_generator(model, tok, strategy)
    cfg = gen_config(strategy, max_steps)
    times, ntoks, accs, passes = [], [], [], 0
    for p in ds:
        r = gen.generate(prompt=p["prompt"], generation_config=cfg)
        times.append(r.total_time); ntoks.append(r.num_tokens_generated)
        a = r.generation_strategy_result.acceptance_rate
        if a is not None:
            accs.append(a)
        ok = run_passk_problem(p["prompt"], r.decoded_prediction, p["test"], p["entry_point"])
        passes += int(ok)
    return summarize(times, ntoks, accs, quality={"pass@1": passes / len(ds)}, n=len(ds))


def eval_cnndm(model, tok, strategy, num, max_steps):
    from torchmetrics.text import ROUGEScore
    examples = get_data(random_shuffle=True, num_samples=num, dataset="cnn_dm_summarization",
                        n_shot=1, seed=SEED, data_path=None, template=None)
    gen = make_generator(model, tok, strategy)
    cfg = gen_config(strategy, max_steps)
    rouge = ROUGEScore(rouge_keys="rouge2")
    times, ntoks, accs = [], [], []
    tok.truncation_side = "left"
    for ex in examples:
        # truncate long article prompts to fit context
        ids = tok(ex.input, return_tensors="pt", truncation=True, max_length=1536).input_ids
        prompt = tok.decode(ids[0], skip_special_tokens=True)
        r = gen.generate(prompt=prompt, generation_config=cfg)
        times.append(r.total_time); ntoks.append(r.num_tokens_generated)
        a = r.generation_strategy_result.acceptance_rate
        if a is not None:
            accs.append(a)
        rouge.update(r.decoded_prediction, ex.output)
    r2 = rouge.compute()["rouge2_fmeasure"].item()
    return summarize(times, ntoks, accs, quality={"rouge2": r2}, n=len(examples))


def summarize(times, ntoks, accs, quality, n):
    # tokens/sec per prompt; mean +/- std across prompts (greedy is deterministic -> across-prompt variance)
    tps = [nt / t for nt, t in zip(ntoks, times) if t > 0 and nt > 0]
    return {
        "n": n,
        "tokens_per_sec_mean": statistics.mean(tps) if tps else None,
        "tokens_per_sec_std": statistics.stdev(tps) if len(tps) > 1 else 0.0,
        "time_per_token_mean": statistics.mean([t / nt for nt, t in zip(ntoks, times) if nt > 0]),
        "acceptance_mean": statistics.mean(accs) if accs else None,
        "quality": quality,
    }


def latency_breakdown(model, tok, n_profile=3, max_steps=64):
    """Profile a few self-SD generations; return top CUDA ops + rough buckets."""
    from torch.profiler import profile, ProfilerActivity
    from datasets import load_dataset
    ds = list(load_dataset("openai_humaneval", split="test"))[:n_profile]
    gen = make_generator(model, tok, "selfsd")
    cfg = gen_config("selfsd", max_steps)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for p in ds:
            gen.generate(prompt=p["prompt"], generation_config=cfg)
    rows = []
    for ev in prof.key_averages():
        cuda = 0
        for attr in ("self_device_time_total", "self_cuda_time_total", "device_time_total", "cuda_time_total"):
            v = getattr(ev, attr, 0) or 0
            if v:
                cuda = v
                break
        if cuda and cuda > 0:
            rows.append((ev.key, cuda))
    rows.sort(key=lambda x: -x[1])
    total = sum(c for _, c in rows) or 1
    top = [{"op": k, "cuda_us": round(c, 1), "pct": round(100 * c / total, 2)} for k, c in rows[:25]]
    # rough buckets by op-name keywords
    def bucket(name):
        n = name.lower()
        if any(k in n for k in ["dequant", "quant", "nf4", "int8", "blockwise", "bnb", "bitsandbytes"]):
            return "dequant"
        if any(k in n for k in ["gemm", "matmul", "mm", "addmm", "linear", "cublas"]):
            return "matmul"
        if any(k in n for k in ["attention", "sdpa", "softmax", "bmm", "flash"]):
            return "attention"
        if any(k in n for k in ["elementwise", "add", "mul", "norm", "rms", "silu", "rotary", "cat", "copy"]):
            return "other_elementwise"
        return "misc"
    buckets = {}
    for k, c in rows:
        buckets[bucket(k)] = buckets.get(bucket(k), 0.0) + c
    buckets_pct = {k: round(100 * v / total, 2) for k, v in sorted(buckets.items(), key=lambda x: -x[1])}
    return {"top_ops": top, "bucket_pct": buckets_pct, "total_cuda_us": round(total, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp16", "nf4", "int8"], required=True)
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=128)
    ap.add_argument("--profile", action="store_true", help="also run latency breakdown (self-SD)")
    ap.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "baselines"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    quantize = None if args.precision == "fp16" else args.precision

    # disk size of the model dir
    disk = sum(os.path.getsize(os.path.join(MODEL, f)) for f in os.listdir(MODEL)
               if f.endswith(".safetensors")) / 1e9

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model, tok = load_model_and_tokenizer(Arguments(model=MODEL, quantize=quantize, seed=SEED))
    load_s = time.time() - t0
    print(f"[load] {args.precision} {load_s:.0f}s | disk {disk:.1f}GB", flush=True)

    # warmup
    wg = make_generator(model, tok, "ar")
    wg.generate(prompt="Hello", generation_config=gen_config("ar", 16))

    results = {"precision": args.precision, "load_s": round(load_s, 1), "disk_gb": round(disk, 2),
               "max_steps": args.max_steps, "configs": {}}
    for strategy in ["ar", "selfsd"]:
        for dataset, fn in [("human_eval", eval_humaneval), ("cnn_dm", eval_cnndm)]:
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            res = fn(model, tok, strategy, args.num, args.max_steps)
            res["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            key = f"{strategy}/{dataset}"
            results["configs"][key] = res
            q = res["quality"]
            print(f"[{args.precision}|{key}] tok/s {res['tokens_per_sec_mean']:.1f}±{res['tokens_per_sec_std']:.1f} | "
                  f"acc {res['acceptance_mean']} | {q} | VRAM {res['vram_peak_gb']}GB | {time.time()-t:.0f}s", flush=True)

    if args.profile:
        print(f"[{args.precision}] latency breakdown (self-SD)...", flush=True)
        results["latency_breakdown"] = latency_breakdown(model, tok)
        print(f"  buckets: {results['latency_breakdown']['bucket_pct']}", flush=True)

    out = os.path.join(args.out_dir, f"baseline_{args.precision}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[save] -> {out}", flush=True)


if __name__ == "__main__":
    main()
