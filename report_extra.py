"""Extra report artifacts from saved preds/margin.

(gamma) gamma-axis break-even: per-position match reconstructed from preds (preds[E]==preds[L]);
        block_accept_rate for gamma in {2,4,6,8} -> acceptance & l_bar per gamma.
        NOTE: speedup(A) needs time/token which depends on gamma -> A is COST-MODEL ESTIMATE
        (c_draft ~ proportional to gamma), not measured. Clearly labeled.
(H) D(E) x token-category heatmap: layer x category disagreement (HumanEval code categories).
(M) margin quantile curve: P(flip | fp16 margin <= x) at E=4 (early) and all-layer pooled.
"""
import json, os, statistics
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
SH = os.path.join(ROOT, "logs", "shadow")
OUT = os.path.join(ROOT, "logs", "report")
os.makedirs(OUT, exist_ok=True)


def block_accept_rate(match, gamma):
    m = [bool(x) for x in match]; n = len(m); pos = tot = acc_t = 0
    while pos < n:
        blk = m[pos:pos + gamma]; a = 0
        for x in blk:
            if x: a += 1
            else: break
        tot += len(blk); acc_t += a
        pos += a + 1   # accepted prefix + 1 verifier token, also on a fully accepted block
    return acc_t / tot if tot else float("nan")


def gen_region(s):
    plen, S = s["prompt_len"], len(s["ids"])
    lo, hi = plen - 1, S - 1
    return (lo, hi) if hi > lo else None


# ---------------- (gamma) gamma-axis acceptance/l_bar ----------------
def gamma_axis():
    fp = torch.load(os.path.join(SH, "preds_fp16.pt"))
    out = {}
    for q in ["nf4", "int8"]:
        pq = torch.load(os.path.join(SH, f"preds_{q}.pt"))
        for g in [2, 4, 6, 8]:
            accs = []
            for sq in pq:
                r = gen_region(sq)
                if not r:
                    continue
                lo, hi = r
                preds = sq["preds"].long()
                match = (preds[7, lo:hi] == preds[-1, lo:hi]).tolist()   # E=8 (index7) as headline
                accs.append(block_accept_rate(match, g))
            a = statistics.mean(accs)
            out.setdefault(q, {})[g] = {"acc_E8": round(a, 4), "l_bar": round(a * g + 1, 3)}
    return out


# ---------------- (H) D(E) x token-category heatmap ----------------
def code_cat(s):
    s = s.strip()
    if s == "": return "other"
    kw = {"def","return","if","else","elif","for","while","in","and","or","not","is","None","True",
          "False","import","from","class","with","as","try","except","lambda","yield","break","continue","pass","raise","assert"}
    if s in kw: return "keyword"
    if (s[0].isalpha() or s[0] == "_") and all(c.isalnum() or c == "_" for c in s): return "identifier"
    if s[0].isdigit() or s[0] in "'\"": return "literal"
    if all(not c.isalnum() and c != "_" for c in s): return "operator"
    return "other"


def heatmap():
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(os.environ.get("QUANTSKIP_MODEL", os.path.expanduser("~/models/layerskip-llama2-7B")))
    fp = torch.load(os.path.join(SH, "preds_fp16.pt"))
    q = torch.load(os.path.join(SH, "preds_nf4.pt"))
    cats = ["keyword", "identifier", "literal", "operator", "other"]
    L = fp[0]["preds"].shape[0]
    sums = {c: np.zeros(L) for c in cats}; cnt = {c: 0 for c in cats}
    for sf, sq in zip(fp, q):
        if sf["ids"] != sq["ids"]:
            continue
        r = gen_region(sf)
        if not r:
            continue
        lo, hi = r
        pf, pq = sf["preds"].long(), sq["preds"].long()
        for t in range(lo, hi):
            c = code_cat(tk.decode([sf["ids"][t + 1]]))
            sums[c] += (pf[:, t] != pq[:, t]).numpy().astype(float); cnt[c] += 1
    mat = np.vstack([sums[c] / max(cnt[c], 1) for c in cats])   # [cats, L]
    plt.figure(figsize=(9, 3.2))
    im = plt.imshow(mat, aspect="auto", cmap="magma", origin="lower")
    plt.colorbar(im, label="D(E) disagreement (nf4 vs fp16)")
    plt.yticks(range(len(cats)), cats); plt.xlabel("layer E (1..32)")
    plt.title("D(E) x token category (HumanEval): early-layer x meaningful-token double concentration")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "de_heatmap.png"), dpi=130); plt.close()
    return {c: round(float((sums[c] / max(cnt[c], 1))[:8].mean()), 4) for c in cats}


# ---------------- (M) margin quantile: P(flip | margin <= x) ----------------
def margin_curve():
    fp = torch.load(os.path.join(SH, "preds_fp16.pt"))
    q = torch.load(os.path.join(SH, "preds_nf4.pt"))
    if "margin" not in fp[0]:
        return None
    rows_E4, rows_all = [], []   # (fp16_margin, flipped?)
    for sf, sq in zip(fp, q):
        if sf["ids"] != sq["ids"]:
            continue
        r = gen_region(sf)
        if not r:
            continue
        lo, hi = r
        pf, pq, mg = sf["preds"].long(), sq["preds"].long(), sf["margin"].float()
        flip = (pf[:, lo:hi] != pq[:, lo:hi]).numpy()   # [L, Lg]
        mm = mg[:, lo:hi].numpy()
        for t in range(flip.shape[1]):
            rows_E4.append((mm[3, t], flip[3, t]))      # E=4 (index3)
            for E in range(flip.shape[0]):
                rows_all.append((mm[E, t], flip[E, t]))

    def curve(rows):
        rows = sorted(rows)
        xs = np.quantile([r[0] for r in rows], np.linspace(0.05, 0.95, 19))
        ys = []
        for x in xs:
            sel = [f for m, f in rows if m <= x]
            ys.append(np.mean(sel) if sel else np.nan)
        return xs, ys
    x4, y4 = curve(rows_E4); xa, ya = curve(rows_all)
    plt.figure(figsize=(6, 4.2))
    plt.plot(x4, y4, "o-", color="C3", label="E=4 (early layer)")
    plt.plot(xa, ya, "s-", color="gray", label="all layers pooled")
    plt.xlabel("fp16 margin threshold x (top1-top2 logit gap)")
    plt.ylabel("P(flip under 4bit | margin <= x)")
    plt.title("Smaller fp16 margin -> higher flip probability (margin causal)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "margin_curve.png"), dpi=130); plt.close()
    # headline: P(flip | margin <= 0.5) at E=4
    lo_sel = [f for m, f in rows_E4 if m <= 0.5]
    hi_sel = [f for m, f in rows_E4 if m > 2.0]
    return {"P_flip_margin<=0.5_E4": round(float(np.mean(lo_sel)), 3) if lo_sel else None,
            "P_flip_margin>2.0_E4": round(float(np.mean(hi_sel)), 3) if hi_sel else None}


def main():
    g = gamma_axis()
    print("== (gamma) acceptance & l_bar by gamma (E=8) [acc=GPU0; A=cost-model] ==")
    for q, gv in g.items():
        print(f"  {q}: " + " | ".join(f"g{k}: acc={v['acc_E8']:.3f} l_bar={v['l_bar']}" for k, v in gv.items()))
    h = heatmap()
    print(f"== (H) D(E<=8) by category (heatmap saved): {h} ==")
    m = margin_curve()
    print(f"== (M) margin curve saved: {m} ==")
    json.dump({"gamma_axis": g, "heatmap_Dearly_by_cat": h, "margin_curve": m},
              open(os.path.join(OUT, "extra.json"), "w"), indent=2)
    print(f"[save] -> {OUT}/(de_heatmap.png, margin_curve.png, extra.json)")


if __name__ == "__main__":
    main()
