"""Shadow-token landscape collection + cross-validation gate.

For one precision (fp16/nf4/int8):
  1) teacher-force a reference sequence (common, --ref-file) through collect_layer_predictions
     to gather per-layer shadow predictions
  2) p_hat(E) = mean_t( layer-E prediction == layer-L prediction ) -> per-layer acceptance landscape
  3) save preds (.pt) -> later D(E) (cross-precision disagreement)
  4) (--cross-val-E) compare with real self-SD acceptance -> cross-validation gate (PASS if |delta|<0.05)

Reference sequence: if --ref-file exists, load it (same sequence across precisions -> paired Delta-alpha / D(E));
otherwise greedy-generate with this model and save.
"""
import argparse, json, os, statistics, time
import torch

from arguments import Arguments
from generate import load_model_and_tokenizer
from data import get_data
from self_speculation.generator_base import GenerationConfig, HuggingfaceLlamaGenerator
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.llama_model_utils import collect_layer_predictions

MODEL = os.environ.get("QUANTSKIP_MODEL", os.path.expanduser("~/models/layerskip-llama2-7B"))
SEED = 42
NUM_SPEC = 6
DEFAULT_NSHOT = {"cnn_dm_summarization": 1, "xsum_summarization": 0, "human_eval": 0}


def block_accept_rate(match_bool, gamma):
    """Simulate self-SD block acceptance from a per-position match sequence (same definition as
    LayerSkip's measured acceptance): propose gamma draft tokens -> accept the longest matching prefix
    (accept-until-first-reject) -> 1 verify bonus token. rate = (accepted draft tokens)/(proposed draft tokens).
    Converts the shadow per-token p_hat into a quantity comparable with measured acceptance.
    """
    m = [bool(x) for x in match_bool]
    n = len(m)
    pos, total_prop, total_acc = 0, 0, 0
    while pos < n:
        block = m[pos: pos + gamma]
        acc = 0
        for x in block:
            if x:
                acc += 1
            else:
                break
        total_prop += len(block)
        total_acc += acc
        # LayerSkip always emits one verifier token after the accepted prefix, including when the
        # whole block is accepted (self_speculation_generator.py: verified_tokens[n_matches]).
        advance = acc + 1
        pos += advance
    return total_acc / total_prop if total_prop else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", choices=["fp16", "nf4", "int8"], required=True)
    ap.add_argument("--dataset", type=str, default="human_eval",
                    help="human_eval | cnn_dm_summarization | xsum_summarization")
    ap.add_argument("--num", type=int, default=30, help="number of samples")
    ap.add_argument("--n-shot", type=int, default=None, help="few-shot count (default per dataset)")
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--max-prompt-tokens", type=int, default=1536,
                    help="left-truncate prompt to this many tokens (avoids context overflow on long articles)")
    ap.add_argument("--ref-file", type=str, default=None,
                    help="reference sequence .pt (generated+saved by this model if missing; else loaded for teacher-forcing)")
    ap.add_argument("--cross-val-E", type=str, default="",
                    help="comma-separated E list (compare with real self-SD). Empty = skip.")
    ap.add_argument("--save-margin", action="store_true",
                    help="also store per-layer top1-top2 logit margin in preds (for margin<->D(E))")
    ap.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "shadow"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    quantize = None if args.precision == "fp16" else args.precision
    n_shot = args.n_shot if args.n_shot is not None else DEFAULT_NSHOT.get(args.dataset, 0)

    examples = get_data(random_shuffle=True, num_samples=args.num, dataset=args.dataset,
                        n_shot=n_shot, seed=SEED, data_path=None, template=None)
    prompts = [ex.input for ex in examples]
    print(f"[data] {len(prompts)} {args.dataset} prompts (n_shot={n_shot})", flush=True)

    t0 = time.time()
    model, tok = load_model_and_tokenizer(Arguments(model=MODEL, quantize=quantize, seed=SEED))
    L = len(model.model.layers)
    print(f"[load] {args.precision} in {time.time()-t0:.0f}s | layers={L} | VRAM {torch.cuda.max_memory_allocated()/1e9:.2f}GB", flush=True)

    # ---- prepare reference sequences (common) ----
    refs = None
    if args.ref_file and os.path.exists(args.ref_file):
        refs = torch.load(args.ref_file)
        print(f"[ref] loaded {len(refs)} sequences from {args.ref_file}", flush=True)
    else:
        refs = []
        tok.truncation_side = "left"   # drop the front (e.g. 1-shot example), keep the target article + "Summary:" cue
        for i, p in enumerate(prompts):
            # left-truncate long prompts (e.g. 1-shot CNN/DM articles) to fit context
            enc = tok(p, return_tensors="pt", add_special_tokens=True,
                      truncation=True, max_length=args.max_prompt_tokens)
            ids = enc.input_ids.to(model.device)
            out = model.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            refs.append({"prompt_len": ids.shape[1], "ids": out[0].tolist()})
            if (i + 1) % 10 == 0:
                print(f"[ref-gen] {i+1}/{len(prompts)}", flush=True)
        if args.ref_file:
            torch.save(refs, args.ref_file)
            print(f"[ref] generated & saved {len(refs)} -> {args.ref_file}", flush=True)

    # ---- shadow collection ----
    # p_curves: per-token match prob p_hat(E)  (clean token-level signal)
    # acc_curves: block acceptance rate (gamma=NUM_SPEC, comparable with real self-SD)
    p_curves, acc_curves = [], []
    saved_preds = []         # for D(E): {"prompt_len","ids","preds"(int32 [L, S])}
    final_match_check = []   # whether layer-L prediction matches the actual next token (generated region) -- sanity
    t1 = time.time()
    for i, r in enumerate(refs):
        ids = torch.tensor([r["ids"]], device=model.device)
        plen, S = r["prompt_len"], len(r["ids"])
        if args.save_margin:
            preds, marg = collect_layer_predictions(model, ids, return_margin=True)  # [L,S],[L,S]
        else:
            preds = collect_layer_predictions(model, ids); marg = None
        verify = preds[-1]                                   # layer-L prediction
        lo, hi = plen - 1, S - 1                              # generated-region positions t: [plen-1, S-2]
        if hi <= lo:
            continue
        reg = slice(lo, hi)
        tgt = ids[0, lo + 1: hi + 1]                          # actual next token (generated region)
        final_match_check.append((verify[reg] == tgt).float().mean().item())
        pmatch = (preds[:, reg] == verify[reg].unsqueeze(0))  # [L, Lg] bool
        p_curves.append(pmatch.float().mean(dim=1).tolist())  # per-token p_hat(E)
        acc_curves.append([block_accept_rate(pmatch[l].tolist(), NUM_SPEC) for l in range(pmatch.shape[0])])
        entry = {"prompt_len": plen, "ids": r["ids"], "preds": preds.to(torch.int32).cpu()}   # int32: int16 would wrap Llama-3 vocab ids (128k)
        if marg is not None:
            entry["margin"] = marg.to(torch.float16).cpu()    # [L, S] top1-top2 logit gap
        saved_preds.append(entry)
        if (i + 1) % 10 == 0:
            print(f"[shadow] {i+1}/{len(refs)} | {time.time()-t1:.0f}s", flush=True)

    # aggregate: per-layer means
    nL = len(p_curves[0])
    p_hat = [statistics.mean(c[e] for c in p_curves) for e in range(nL)]          # per-token
    acc_hat = [statistics.mean(c[e] for c in acc_curves) for e in range(nL)]      # block acceptance
    print(f"[shadow] done {len(p_curves)} samples | {time.time()-t1:.0f}s", flush=True)
    print(f"[sanity] layer-L vs actual next-token match (generated region, expect ~1.0): {statistics.mean(final_match_check):.3f}", flush=True)

    torch.save(saved_preds, os.path.join(args.out_dir, f"preds_{args.precision}.pt"))

    # ---- cross-validation gate: real self-SD acceptance ----
    cross = {}
    Es = [int(x) for x in args.cross_val_E.split(",") if x.strip()]
    if Es:
        gen = HuggingfaceLlamaGenerator(tokenizer=tok, model=model,
                                        generation_strategy=SelfSpeculativeGenerationStrategy())
        for E in Es:
            accs = []
            for p in prompts:
                cfg = GenerationConfig(max_steps=args.gen_tokens, exit_layer=E, num_speculations=NUM_SPEC,
                                       generation_strategy="self_speculative", sample=False,
                                       stop_words=[], stop_token_ids=[])
                r = gen.generate(prompt=p, generation_config=cfg)
                a = r.generation_strategy_result.acceptance_rate
                if a is not None:
                    accs.append(a)
            real = statistics.mean(accs) if accs else float("nan")
            shadow = acc_hat[E - 1]                            # compare with block acceptance (same definition)
            diff = abs(real - shadow)
            cross[E] = {"real_selfSD": real, "shadow_blockrate": shadow,
                        "shadow_pertoken": p_hat[E - 1], "abs_diff": diff, "pass": diff < 0.05}
            print(f"[cross-val] E={E:2d} | real {real:.3f} vs shadow(block) {shadow:.3f} "
                  f"(p_hat={p_hat[E-1]:.3f}) | |delta|={diff:.3f} | {'PASS' if diff<0.05 else 'FAIL'}", flush=True)

    summary = {
        "precision": args.precision, "n": len(p_curves), "num_layers": nL,
        "gen_tokens": args.gen_tokens, "num_speculations": NUM_SPEC,
        "p_hat_per_layer": p_hat,            # per-token match prob
        "acc_hat_per_layer": acc_hat,        # block acceptance rate (for real comparison)
        "final_layer_match_mean": statistics.mean(final_match_check),
        "cross_val": cross,
    }
    with open(os.path.join(args.out_dir, f"shadow_{args.precision}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] -> {args.out_dir}/shadow_{args.precision}.json (+ preds_{args.precision}.pt)", flush=True)
    idx = [0, 3, 7, 11, 15, 23, nL - 1]
    print(f"[p_hat(E) per-token] { {e+1: round(p_hat[e],3) for e in idx} }", flush=True)
    print(f"[acc(E) block]       { {e+1: round(acc_hat[e],3) for e in idx} }", flush=True)


if __name__ == "__main__":
    main()
