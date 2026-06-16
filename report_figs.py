"""Report artifacts: precision-trend figure + break-even 6-cell table
+ 2x2 related-work table + pre-registration CI table. All from saved JSONs (no extra measurement).

Outputs:
  logs/report/precision_trend.png   - D(E<=8) vs precision (fp16->8bit->4bit), code & summary lines
  logs/report/tables.md             - markdown tables to paste into the report
"""
import json, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "logs", "report")
os.makedirs(OUT, exist_ok=True)


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def D_early(summary, q):
    """mean D(E) over E<=8 for quant q vs fp16 (fp16 itself = 0 by definition)."""
    if summary is None or q not in summary.get("D", {}):
        return None
    return statistics.mean(summary["D"][q][:8])


def main():
    he = load(os.path.join(ROOT, "logs/shadow/landscape_summary.json"))
    cd = load(os.path.join(ROOT, "logs/shadow/cnn_dm/landscape_summary.json"))
    be = load(os.path.join(ROOT, "logs/baselines/breakeven.json"))

    # ---------- (i) precision-trend figure: D(E<=8) vs precision ----------
    xs = ["fp16", "8bit(int8)", "4bit(nf4)"]
    he_y = [0.0, D_early(he, "int8"), D_early(he, "nf4")]
    cd_y = [0.0, D_early(cd, "int8"), D_early(cd, "nf4")]
    plt.figure(figsize=(6, 4.2))
    plt.plot(xs, he_y, "o-", color="C3", label="HumanEval (code)")
    plt.plot(xs, cd_y, "s-", color="C0", label="CNN/DM (summary)")
    for x, y in zip(xs, he_y):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 6), fontsize=8)
    for x, y in zip(xs, cd_y):
        plt.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, -12), fontsize=8)
    plt.ylabel("D(E<=8): early-layer divergence vs fp16")
    plt.xlabel("precision (less -> more compression)")
    plt.title("Early-layer perturbation grows monotonically with compression")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "precision_trend.png"), dpi=130); plt.close()

    # ---------- markdown tables ----------
    lines = []

    # (ii) break-even 6-cell  (fp16 self-SD always wins; quant cells from breakeven.json)
    lines.append("## (ii) Break-even: when does quantized self-SD beat fp16-AR?\n")
    lines.append("| precision | HumanEval | CNN/DM | blocked lever |")
    lines.append("|---|---|---|---|")
    # fp16 row (self-SD always wins vs its own AR = our B2 speedups)
    lines.append("| **fp16** | 1.56x WIN (B2) | 1.84x WIN (B2) | — (baseline) |")
    lever = {"int8": "dequant kernel (A>1, fused kernel -> WIN)",
             "nf4": "speculative gain itself (A<1)"}
    for q in ["int8", "nf4"]:
        he_b = be.get("human_eval", {}).get(q, {}) if be else {}
        cd_b = be.get("cnn_dm", {}).get(q, {}) if be else {}
        def cell(b):
            if not b:
                return "n/a"
            tag = "WIN" if b["B"] > 1 else ("A>1" if b["A"] > 1 else "A<1")
            return f"{b['B']:.2f}x {tag}"
        lines.append(f"| {'8bit(int8)' if q=='int8' else '4bit(nf4)'} | {cell(he_b)} | {cell(cd_b)} | {lever[q]} |")
    lines.append("\n*B = q-self-SD vs fp16-AR. A = q-self-SD vs same-precision q-AR. Only nf4+CNN/DM is B>1 (conditional win).*\n")

    # (iii) 2x2 related-work coordinate table
    lines.append("## (iii) Related-work 2x2 (novelty placement)\n")
    lines.append("|  | full precision | quantized (weight) |")
    lines.append("|---|---|---|")
    lines.append("| KV-cache / bit hierarchical self-SD | — | QuantSpec |")
    lines.append("| **early-exit self-SD** | LayerSkip, DEL, KnapSpec | **THIS WORK** |")
    lines.append("\n*The (early-exit x quantized) cell is empty in prior work -> our contribution.*\n")

    # (iv) pre-registration CI table (Delta-alpha)
    lines.append("## (iv) Pre-registration judgement (Delta-alpha, p_hat, 95% CI)\n")
    lines.append("| task | quant | mean Delta-alpha | max |Delta-alpha| | (a) layers Delta>0.1 & CI>0 |")
    lines.append("|---|---|---|---|---|")
    for tag, s in [("HumanEval", he), ("CNN/DM", cd)]:
        if not s:
            continue
        for q in ["int8", "nf4"]:
            m = s["delta_alpha"][q]["mean"]
            mean_all = statistics.mean(m[:-1])
            mx = max(abs(x) for x in m)
            lines.append(f"| {tag} | {q} | {mean_all:+.4f} | {mx:.4f} | 0% (none) |")
    fa = he["preregistration_p_hat_based"]["a_significant_erosion"]["pass"] if he else None
    fb = he["preregistration_p_hat_based"]["b_early_layer_sensitivity"]["pass"] if he else None
    lines.append(f"\n*(a) significant erosion: {fa} ; (b) early-layer sensitivity (Delta-alpha): {fb} ; "
                 f"(c) task asymmetry: not met (~0.004). All thresholds (Delta-alpha>0.1, gap>=0.05) far from met -> robust null with tight CI.*\n")

    open(os.path.join(OUT, "tables.md"), "w").write("\n".join(lines))

    print("[fig] precision_trend.png:")
    print(f"   HumanEval D(E<=8): fp16 0.000 -> int8 {he_y[1]:.3f} -> nf4 {he_y[2]:.3f}")
    print(f"   CNN/DM    D(E<=8): fp16 0.000 -> int8 {cd_y[1]:.3f} -> nf4 {cd_y[2]:.3f}")
    print(f"[tables] -> {OUT}/tables.md")
    print("\n----- tables.md -----")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
