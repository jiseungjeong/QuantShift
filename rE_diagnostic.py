"""QuantShift AED go/no-go diagnostic: measure r(E) = c_draft(E)/c_verify on the REAL self-SD path.

AED (Adaptive Exit Depth) bets that lowering draft exit layer E reduces draft cost ~proportionally
(c_draft ~ E layers), so per-token depth tuning shrinks r and pushes A>1. But in the actual LayerSkip
loop the draft phase is `num_speculations` (=gamma) SEQUENTIAL single-token forward_early() calls
(each batch=seq=1, memory-bandwidth-bound: weight streaming + kernel launch dominate), while verify is
ONE batched forward_remainder() call. If decode latency is launch/bandwidth-bound, halving E barely
moves c_draft -> r is roughly FLAT in E -> AED's r-lever is weak and we should not invest 1-2 days.

This times, for nf4 on a real prompt with a real KV cache (mirroring single_step_speculation):
  - c_draft(E) = wall-time of gamma sequential forward_early() at exit_layer=E (the real draft phase)
  - c_verify   = wall-time of one forward_remainder() over the gamma-token block
  - r(E) = (c_draft(E)/gamma) / c_verify   [per-draft-token cost vs one verify call]
Decision: if r(E) scales ~linearly with E (r(4) << r(8)) -> AED viable, proceed. If r(E) ~flat
(r(4) ~= r(8)) -> draft cost is launch/bandwidth-bound, AED's depth lever is weak -> kill, report.

Run: cd LayerSkip && conda run -n NLP env PYTHONNOUSERSITE=1 PYTHONPATH=$PWD python ../rE_diagnostic.py
"""
import os, time, json
import torch

from arguments import Arguments
from generate import load_model_and_tokenizer
from data import get_data
from self_speculation.llama_model_utils import forward_early, forward_remainder

MODEL = os.path.expanduser("~/models/layerskip-llama2-7B")
SEED = 42
GAMMA = 6
E_LIST = [2, 4, 6, 8, 12, 16]
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "logs", "report")


def time_call(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def draft_phase(model, prompt_ids, E):
    """gamma sequential single-token forward_early calls (the real draft phase). Returns nothing;
    builds its own throwaway KV so repeated timing is independent."""
    # prefill the prompt through E layers once (sets up KV), then gamma decode steps
    pkv = None
    exit_cache = None
    cur = prompt_ids
    for step in range(GAMMA + 1):   # +1: first call is prefill (prompt), then gamma decode steps
        res = forward_early(model, cur, pkv, E, exit_cache)
        pkv = res.past_key_values
        exit_cache = res.exit_query_cache
        nxt = res.logits[:, -1, :].argmax(-1, keepdim=True).int()
        cur = nxt


def main():
    os.makedirs(OUT, exist_ok=True)
    examples = get_data(random_shuffle=True, num_samples=4, dataset="human_eval",
                        n_shot=0, seed=SEED, data_path=None, template=None)
    model, tok = load_model_and_tokenizer(Arguments(model=MODEL, quantize="nf4", seed=SEED))
    L = len(model.model.layers)
    print(f"[load] nf4 | layers={L} | gamma={GAMMA}", flush=True)

    enc = tok(examples[0].input, return_tensors="pt", truncation=True, max_length=512)
    prompt_ids = enc.input_ids.to(model.device).int()
    print(f"[prompt] {prompt_ids.shape[1]} tokens", flush=True)

    # --- c_draft(E): the REAL draft phase = gamma sequential single-token forward_early at E ---
    # AED's whole bet is that c_draft scales with E. We test that DIRECTLY (no verify needed:
    # the AED-relevant question is "does lowering E cut the per-draft-token decode cost?").
    # Also time a single decode-step forward_early at E (the per-token primitive AED would shorten).
    rows = []
    for E in E_LIST:
        # whole draft phase (prefill + gamma decodes), as in single_step_speculation
        c_phase = time_call(lambda: draft_phase(model, prompt_ids, E), iters=5)
        # isolated single decode step at E (1 token, with a 1-step KV already present)
        warm = forward_early(model, prompt_ids, None, E, None)
        pkv0, ec0 = warm.past_key_values, warm.exit_query_cache
        one_tok = warm.logits[:, -1, :].argmax(-1, keepdim=True).int()

        def one_decode():
            forward_early(model, one_tok, pkv0, E, ec0)
        c_step = time_call(one_decode, iters=20)
        rows.append({"E": E, "draft_phase_ms": round(c_phase*1000, 2),
                     "one_decode_step_ms": round(c_step*1000, 3)})
        print(f"[draft] E={E:2d} | phase(prefill+{GAMMA}dec) {c_phase*1000:7.2f} ms | "
              f"1 decode step {c_step*1000:6.3f} ms", flush=True)

    # --- verdict: does the per-decode-step cost scale with E? ---
    def step(E): return next(x["one_decode_step_ms"] for x in rows if x["E"] == E)
    s2, s4, s8, s16 = step(2), step(4), step(8), step(16)
    # linear-in-E: step(4)/step(8) ~ 0.5 ; flat (launch/bandwidth bound): ~1.0
    ratio_4_8 = s4 / s8
    ratio_2_8 = s2 / s8
    print(f"\n==================== r(E) / draft-cost scaling VERDICT ====================")
    print(f"  1-decode-step ms: E2={s2:.3f} E4={s4:.3f} E8={s8:.3f} E16={s16:.3f}")
    print(f"  step(4)/step(8)={ratio_4_8:.3f}  (linear-in-E ~0.50 | flat/launch-bound ~1.0)")
    print(f"  step(2)/step(8)={ratio_2_8:.3f}  (linear-in-E ~0.25 | flat ~1.0)")
    if ratio_4_8 < 0.7:
        print(f"  >>> AED VIABLE: draft decode cost scales with E -> lowering E cuts c_draft -> r down. Proceed.")
        verdict = "AED_VIABLE"
    elif ratio_4_8 > 0.85:
        print(f"  >>> AED WEAK: draft decode cost ~FLAT in E (launch/bandwidth-bound) -> depth lever barely "
              f"moves r. Kill AED; real lever is gamma (sequential-call count), not depth.")
        verdict = "AED_WEAK"
    else:
        print(f"  >>> AED MARGINAL: partial E-scaling. Modest help possible; weigh vs deadline.")
        verdict = "AED_MARGINAL"

    json.dump({"gamma": GAMMA, "rows": rows, "ratio_step4_step8": round(ratio_4_8, 3),
               "ratio_step2_step8": round(ratio_2_8, 3), "verdict": verdict},
              open(os.path.join(OUT, "rE_diagnostic.json"), "w"), indent=2)
    print(f"[save] -> {OUT}/rE_diagnostic.json")


if __name__ == "__main__":
    main()
