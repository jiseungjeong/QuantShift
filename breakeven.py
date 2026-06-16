"""Break-even analysis for quantized self-SD (no extra measurement).

Pulls tokens/sec + acceptance from logs/baselines/baseline_{fp16,nf4,int8}.json and reports, per dataset & quant:
  - A = speedup of q-self-SD vs SAME-precision q-AR  (pure speculative gain)
  - threshold = tps(fp16-AR) / tps(q-AR)             (the AR-level slowdown factor q must overcome)
  - B = speedup of q-self-SD vs fp16-AR              (deployment view; our ~0.84x result)
  Identity: B = A / threshold. So q-self-SD beats fp16-AR (B>1)  <=>  A > threshold.
  l_bar = acceptance * gamma + 1                     (mean tokens advanced per verify step; info)

Interpretation lever: a faster dequant kernel shrinks `threshold` (toward 1 if q-AR matched fp16-AR);
even then B>1 needs A>1, i.e. self-SD must actually beat same-precision AR.
"""
import json, os

BDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "baselines")
GAMMA = 6


def tps(d, prec, strat, dataset):
    try:
        return d[prec]["configs"][f"{strat}/{dataset}"]["tokens_per_sec_mean"]
    except (KeyError, TypeError):
        return None


def acc(d, prec, dataset):
    try:
        return d[prec]["configs"][f"selfsd/{dataset}"]["acceptance_mean"]
    except (KeyError, TypeError):
        return None


def main():
    d = {}
    for p in ["fp16", "nf4", "int8"]:
        f = os.path.join(BDIR, f"baseline_{p}.json")
        if os.path.exists(f):
            d[p] = json.load(open(f))
    if "fp16" not in d:
        print(f"[err] need baseline_fp16.json in {BDIR}"); return

    out = {}
    for dataset in ["human_eval", "cnn_dm"]:
        fp_ar = tps(d, "fp16", "ar", dataset)
        print(f"\n==================== BREAK-EVEN — {dataset} (gamma={GAMMA}) ====================")
        print(f"fp16-AR baseline: {fp_ar:.1f} tok/s")
        print(f"{'quant':6} {'l_bar':>6} {'A(vs q-AR)':>11} {'threshold':>10} {'B(vs fp16-AR)':>13} {'verdict':>10}")
        for q in ["int8", "nf4"]:
            if q not in d:
                continue
            q_ar = tps(d, q, "ar", dataset)
            q_ssd = tps(d, q, "selfsd", dataset)
            a = acc(d, q, dataset)
            if None in (q_ar, q_ssd, a):
                continue
            A = q_ssd / q_ar                      # self-SD vs same-precision AR
            threshold = fp_ar / q_ar              # A must exceed this for B>1
            B = q_ssd / fp_ar                     # self-SD vs fp16 AR
            lbar = a * GAMMA + 1
            verdict = "B>1 WIN" if B > 1 else ("A>1 (q-SD>q-AR)" if A > 1 else "A<1 (SD hurts)")
            out.setdefault(dataset, {})[q] = {"l_bar": round(lbar, 2), "A": round(A, 3),
                                              "threshold": round(threshold, 3), "B": round(B, 3),
                                              "verdict": verdict}
            print(f"{q:6} {lbar:>6.2f} {A:>11.3f} {threshold:>10.3f} {B:>13.3f} {verdict:>16}")

        # what would it take? (lower dequant tax so q-AR matches fp16-AR -> threshold->1)
        print("  -- if a fused kernel made q-AR as fast as fp16-AR (threshold->1): B -> A --")
        for q in ["int8", "nf4"]:
            r = out.get(dataset, {}).get(q)
            if r:
                need = "already A>1 -> would WIN" if r["A"] > 1 else f"still B=A={r['A']:.2f}<1 -> self-SD must beat q-AR first"
                print(f"     {q}: B would become A={r['A']:.2f}  ({need})")

    json.dump(out, open(os.path.join(BDIR, "breakeven.json"), "w"), indent=2)
    print(f"\n[note] A=q-self-SD vs q-AR; threshold=fp16AR/qAR (=AR-level dequant slowdown); B=A/threshold=q-self-SD vs fp16-AR.")
    print(f"[save] -> {BDIR}/breakeven.json")


if __name__ == "__main__":
    main()
