"""QuantShift §4 fix: recompute HumanEval pass@1 after the indentation-decode fix.

Only AR is needed: with greedy decoding, self-SD output is identical to AR output at the same
precision (self-SD verifies/falls back to the full model's greedy token), so pass@1(self-SD)=pass@1(AR).
Saves completions + per-problem pass so re-scoring later is free.

Run: cd LayerSkip && conda run -n NLP env PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python ../rescore_passk.py --num 100
"""
import argparse, json, os, time
import torch
from arguments import Arguments
from generate import load_model_and_tokenizer
from self_speculation.generator_base import GenerationConfig, HuggingfaceLlamaGenerator
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from datasets import load_dataset
from baseline_eval import run_passk_problem, fix_indent, truncate_completion

MODEL = os.path.expanduser("~/models/layerskip-llama2-7B")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "baselines")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=128)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ds = list(load_dataset("openai_humaneval", split="test"))[:args.num]
    results = {}
    for prec in ["fp16", "nf4", "int8"]:
        quantize = None if prec == "fp16" else prec
        model, tok = load_model_and_tokenizer(Arguments(model=MODEL, quantize=quantize, seed=42))
        gen = HuggingfaceLlamaGenerator(tokenizer=tok, model=model,
                                        generation_strategy=AutoRegressiveGenerationStrategy())
        cfg = GenerationConfig(max_steps=args.max_steps, generation_strategy="autoregressive",
                               sample=False, stop_words=[], stop_token_ids=[])
        passes, comps = 0, []
        t0 = time.time()
        for p in ds:
            r = gen.generate(prompt=p["prompt"], generation_config=cfg)
            ok = run_passk_problem(p["prompt"], r.decoded_prediction, p["test"], p["entry_point"])
            passes += int(ok)
            comps.append({"task_id": p["task_id"], "pass": ok,
                          "completion": truncate_completion(fix_indent(r.decoded_prediction))})
        rate = passes / len(ds)
        results[prec] = {"pass@1": rate, "n": len(ds)}
        json.dump({"pass@1": rate, "n": len(ds), "completions": comps},
                  open(os.path.join(OUT, f"passk_{prec}.json"), "w"), indent=2)
        print(f"[{prec}] pass@1 = {rate:.3f} ({passes}/{len(ds)}) | {time.time()-t0:.0f}s", flush=True)
        del model, gen
        import gc; gc.collect(); torch.cuda.empty_cache()
    print("PASSK:", {k: v["pass@1"] for k, v in results.items()}, flush=True)


if __name__ == "__main__":
    main()
