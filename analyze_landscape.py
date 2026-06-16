"""Compute Delta-alpha / D(E) / precision trend from saved shadow preds,
run the pre-registered judgement, and plot.

Prerequisite: run shadow_landscape.py for fp16/nf4/int8 each with the SAME ref_common.pt, so that
`logs/shadow/preds_{precision}.pt` exist (paired on identical sequences). Each entry: {prompt_len, ids, preds[L,S] int16}.

Metrics (locked decisions):
  - alpha = p_hat(E) per-token match prob (= layer-E argmax == layer-L argmax, averaged over generated region)
  - delta_alpha_q(E) = p_hat_fp16(E) - p_hat_q(E)   (paired, per-sample -> 95% CI)
  - D_q(E)          = mean_t[ preds_fp16[E,t] != preds_q[E,t] ]  (top-1 disagreement, common sequence)
Pre-registration (p_hat-based): (a) Delta-alpha>0.1 & 95%CI excludes 0 over >= half the layers ;
                                (b) mean Delta-alpha(E<=8) - mean Delta-alpha(E>=16) >= 0.05.
"""
import argparse, json, os, math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = "fp16"                 # full-precision reference
QUANTS = ["int8", "nf4"]     # 8bit, 4bit (in trend order)

_ap = argparse.ArgumentParser()
_ap.add_argument("--shadow-dir", type=str,
                 default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "shadow"),
                 help="dir holding preds_<precision>.pt (per-dataset, e.g. logs/shadow or logs/shadow/cnn_dm)")
_ap.add_argument("--tag", type=str, default="HumanEval", help="dataset tag for titles/output")
_ARGS = _ap.parse_args()
SHADOW_DIR = _ARGS.shadow_dir
TAG = _ARGS.tag
FIG_DIR = os.path.join(SHADOW_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)


def load_preds(prec):
    path = os.path.join(SHADOW_DIR, f"preds_{prec}.pt")
    if not os.path.exists(path):
        return None
    return torch.load(path)


def per_sample_pmatch(samples):
    """For each sample -> [L] per-token match rate (p_hat). Also return per-layer argmax over the
    generated region (used for D(E))."""
    p_rows, regions = [], []
    for s in samples:
        preds = s["preds"].long()           # [L, S]
        L, S = preds.shape
        plen = s["prompt_len"]
        lo, hi = plen - 1, S - 1
        if hi <= lo:
            p_rows.append(None); regions.append(None); continue
        verify = preds[-1, lo:hi]            # [Lg]
        seg = preds[:, lo:hi]                # [L, Lg]
        p_rows.append((seg == verify.unsqueeze(0)).float().mean(dim=1).numpy())  # [L]
        regions.append(seg.numpy())          # [L, Lg] argmax ids (for D(E))
    return p_rows, regions


def mean_ci(arr):  # arr: [n, L] -> mean[L], ci_halfwidth[L] (95%)
    a = np.array(arr, dtype=float)
    m = a.mean(0)
    sd = a.std(0, ddof=1)
    n = a.shape[0]
    half = 1.96 * sd / math.sqrt(n)
    return m, half


def main():
    data = {p: load_preds(p) for p in [REF] + QUANTS}
    missing = [p for p, v in data.items() if v is None]
    if missing:
        print(f"[err] missing preds: {missing}. Run shadow_landscape.py with --ref-file ref_common.pt first.", flush=True)
        return

    # Paired consistency: with the same ref, sample count and ids must match.
    n = min(len(data[p]) for p in data)
    p_rows, regions = {}, {}
    for p in data:
        pr, rg = per_sample_pmatch(data[p][:n])
        p_rows[p], regions[p] = pr, rg

    # Keep only samples valid across all precisions (region exists & identical ids = truly common ref).
    valid = []
    for i in range(n):
        if any(p_rows[p][i] is None for p in data):
            continue
        if any(data[p][i]["ids"] != data[REF][i]["ids"] for p in data):
            continue   # not a common ref -> cannot pair -> skip
        valid.append(i)
    print(f"[data] paired valid samples: {len(valid)}/{n}", flush=True)
    L = len(p_rows[REF][valid[0]])

    # p_hat(E) per precision (sample mean)
    p_hat = {p: np.array([p_rows[p][i] for i in valid]).mean(0) for p in data}  # {p:[L]}

    # delta_alpha_q(E) (per-sample paired diff -> mean + CI)
    delta = {}
    for q in QUANTS:
        per = np.array([p_rows[REF][i] - p_rows[q][i] for i in valid])  # [nv, L]
        m, half = mean_ci(per)
        delta[q] = {"mean": m, "ci": half}

    # D_q(E) top-1 disagreement (common sequence, sample mean)
    Dcurve = {}
    for q in QUANTS:
        per = []
        for i in valid:
            rf, rq = regions[REF][i], regions[q][i]   # [L, Lg]
            per.append((rf != rq).mean(axis=1))        # [L]
        Dcurve[q] = np.array(per).mean(0)              # [L]

    Es = np.arange(1, L + 1)

    # ---- pre-registration judgement (p_hat-based, 4bit=nf4) ----
    nf4d = delta["nf4"]["mean"]; nf4ci = delta["nf4"]["ci"]
    sig = (nf4d > 0.1) & ((nf4d - nf4ci) > 0)          # Delta-alpha>0.1 & CI lower bound > 0
    frac_sig = sig[:L - 1].mean()                       # exclude E=L
    a_pass = frac_sig >= 0.5
    early = nf4d[:8].mean(); late = nf4d[15:].mean()    # E<=8 vs E>=16
    b_gap = early - late; b_pass = b_gap >= 0.05
    # precision trend: fraction of E where Delta-alpha_4bit >= Delta-alpha_8bit
    trend_frac = (delta["nf4"]["mean"] >= delta["int8"]["mean"])[:L - 1].mean()
    # D(E) early vs late (mechanism)
    D_early = {q: Dcurve[q][:8].mean() for q in QUANTS}
    D_late = {q: Dcurve[q][15:].mean() for q in QUANTS}

    # ---- plots ----
    # 1) acceptance curves
    plt.figure(figsize=(7, 4.5))
    for p, c in [(REF, "k"), ("int8", "C0"), ("nf4", "C3")]:
        plt.plot(Es, p_hat[p], marker=".", label=rf"$\hat{{p}}(E)$  {p}", color=c)
    plt.xlabel("exit layer $E$"); plt.ylabel(r"$\hat{p}(E)$  per-token acceptance"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "acceptance_curves.png"), dpi=130); plt.close()

    # 2) delta_alpha(E) + CI
    plt.figure(figsize=(7, 4.5))
    for q, c in [("int8", "C0"), ("nf4", "C3")]:
        m, h = delta[q]["mean"], delta[q]["ci"]
        plt.plot(Es, m, marker=".", label=rf"$\Delta\alpha(E)$  {q}", color=c)
        plt.fill_between(Es, m - h, m + h, alpha=.2, color=c)
    plt.axhline(0, color="gray", lw=.8); plt.axhline(0.1, color="r", ls="--", lw=.8, label="threshold 0.1")
    plt.xlabel("exit layer $E$"); plt.ylabel(r"$\Delta\alpha(E)=\hat{p}_{\mathrm{fp16}}(E)-\hat{p}_{q}(E)$"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "delta_alpha.png"), dpi=130); plt.close()

    # 3) D(E)
    plt.figure(figsize=(7, 4.5))
    for q, c in [("int8", "C0"), ("nf4", "C3")]:
        plt.plot(Es, Dcurve[q], marker=".", label=rf"$D(E)$  {q}", color=c)
    plt.xlabel("exit layer $E$"); plt.ylabel(r"$D(E)$  top-1 disagreement (vs fp16)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "divergence.png"), dpi=130); plt.close()

    # ---- summary save/print ----
    summary = {
        "n_valid": len(valid),
        "p_hat": {p: p_hat[p].round(4).tolist() for p in data},
        "delta_alpha": {q: {"mean": delta[q]["mean"].round(4).tolist(),
                            "ci95": delta[q]["ci"].round(4).tolist()} for q in QUANTS},
        "D": {q: Dcurve[q].round(4).tolist() for q in QUANTS},
        "preregistration_p_hat_based": {
            "a_significant_erosion": {"frac_layers_delta>0.1_and_CI": round(float(frac_sig), 3), "pass": bool(a_pass)},
            "b_early_layer_sensitivity": {"mean_delta(E<=8)": round(float(early), 4),
                                          "mean_delta(E>=16)": round(float(late), 4),
                                          "gap": round(float(b_gap), 4), "pass": bool(b_pass)},
        },
        "precision_trend_frac_delta4bit_ge_delta8bit": round(float(trend_frac), 3),
        "D_early_vs_late": {q: {"E<=8": round(float(D_early[q]), 4), "E>=16": round(float(D_late[q]), 4)} for q in QUANTS},
    }
    with open(os.path.join(SHADOW_DIR, "landscape_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n==================== LANDSCAPE SUMMARY ({TAG}) ====================", flush=True)
    idx = [0, 3, 7, 11, 15, 19, 23, L - 1]
    print(f"E:               {[int(Es[e]) for e in idx]}", flush=True)
    print(f"p_hat fp16:      {[round(p_hat['fp16'][e],3) for e in idx]}", flush=True)
    print(f"p_hat int8(8b):  {[round(p_hat['int8'][e],3) for e in idx]}", flush=True)
    print(f"p_hat nf4(4b):   {[round(p_hat['nf4'][e],3) for e in idx]}", flush=True)
    print(f"delta_a 4bit:    {[round(delta['nf4']['mean'][e],3) for e in idx]}", flush=True)
    print(f"delta_a 8bit:    {[round(delta['int8']['mean'][e],3) for e in idx]}", flush=True)
    print(f"D(E) 4bit:       {[round(Dcurve['nf4'][e],3) for e in idx]}", flush=True)
    print(f"D(E) 8bit:       {[round(Dcurve['int8'][e],3) for e in idx]}", flush=True)
    print("---- pre-registration judgement (p_hat-based, 4bit) ----", flush=True)
    print(f"(a) significant erosion: frac of layers with delta>0.1 & CI>0 = {frac_sig:.2f} -> {'PASS' if a_pass else 'not met'}", flush=True)
    print(f"(b) early-layer sensitivity: mean_delta(E<=8) {early:.3f} - mean_delta(E>=16) {late:.3f} = {b_gap:+.3f} (>=0.05?) -> {'PASS' if b_pass else 'not met'}", flush=True)
    print(f"precision trend: frac of E with delta_4bit >= delta_8bit = {trend_frac:.2f}", flush=True)
    print(f"D(E) early vs late: 4bit E<=8={D_early['nf4']:.3f} vs E>=16={D_late['nf4']:.3f}", flush=True)
    print(f"[plots] {FIG_DIR}/ (acceptance_curves, delta_alpha, divergence).png", flush=True)
    print("======================================================================", flush=True)


if __name__ == "__main__":
    main()
