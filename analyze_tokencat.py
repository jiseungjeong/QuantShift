"""Per-token-category breakdown of D(E) + margin<->D(E) correlation.

Uses saved shadow preds (paired on a common fp16 reference):
  - D(E) per token category: for each generated position, label the *target token* by category, then
    D_cat(E) = mean over positions in that category of [ preds_fp16[E,t] != preds_q[E,t] ].
    HumanEval (code): python tokenize -> keyword / identifier / operator / literal / other.
    CNN/DM (summary): spaCy POS -> function-word (DET/ADP/CCONJ/PRON/AUX/PART/SCONJ) vs content-word (NOUN/VERB/ADJ/PROPN/NUM/ADV).
  - margin<->D(E): if preds_fp16 has 'margin' (top1-top2 logit gap), show that small fp16 margin
    correlates with high D(E) at the same (layer, position) = "close call -> flipped by quant noise".
"""
import argparse, io, json, os, tokenize as pytok
import numpy as np
import torch

REF = "fp16"
QUANT = "nf4"   # 4bit as the headline quant for the split


def code_category(tok_str):
    s = tok_str.strip()
    if s == "":
        return "other"
    if s in {"def","return","if","else","elif","for","while","in","and","or","not","is","None","True",
             "False","import","from","class","with","as","try","except","lambda","yield","break","continue",
             "pass","raise","assert","global","del","while"}:
        return "keyword"
    if s and (s[0].isalpha() or s[0] == "_") and all(c.isalnum() or c == "_" for c in s):
        return "identifier"
    if s[0].isdigit() or (s[0] in "'\"") or s in {"True", "False", "None"}:
        return "literal"
    if all(not c.isalnum() and c != "_" for c in s):
        return "operator"
    return "other"


def load(shadow_dir, prec):
    p = os.path.join(shadow_dir, f"preds_{prec}.pt")
    return torch.load(p) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shadow-dir", required=True)
    ap.add_argument("--tag", default="HumanEval")
    ap.add_argument("--kind", choices=["code", "pos"], required=True)
    args = ap.parse_args()

    fp = load(args.shadow_dir, REF)
    q = load(args.shadow_dir, QUANT)
    if fp is None or q is None:
        print(f"[err] need preds_{REF}.pt and preds_{QUANT}.pt in {args.shadow_dir}"); return

    # tokenizer to map ids -> token strings (for labeling target tokens)
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(os.environ.get("QUANTSKIP_MODEL", os.path.expanduser("~/models/layerskip-llama2-7B")))
    nlp = None
    if args.kind == "pos":
        import spacy
        nlp = spacy.load("en_core_web_sm")
        FUNC = {"DET", "ADP", "CCONJ", "PRON", "AUX", "PART", "SCONJ"}
        CONTENT = {"NOUN", "VERB", "ADJ", "PROPN", "NUM", "ADV"}

    L = fp[0]["preds"].shape[0]
    # accumulate disagreement counts per category per layer
    cats = {}
    has_margin = "margin" in fp[0]
    margin_flip, margin_same = [], []   # fp16 margin where 4bit flipped vs not (at layer E_probe)
    E_probe = 3  # an early layer (1-based 4) for margin illustration

    for sf, sq in zip(fp, q):
        if sf["ids"] != sq["ids"]:
            continue
        plen, S = sf["prompt_len"], len(sf["ids"])
        lo, hi = plen - 1, S - 1
        if hi <= lo:
            continue
        ids = sf["ids"]
        pf = sf["preds"].long(); pq = sq["preds"].long()
        marg = sf["margin"].float() if has_margin else None

        # label each generated position by its target token category
        for t in range(lo, hi):
            tgt_id = ids[t + 1]
            tok_str = tk.decode([tgt_id])
            if args.kind == "code":
                cat = code_category(tok_str)
            else:
                doc = nlp(tok_str.strip()) if tok_str.strip() else None
                pos = doc[0].pos_ if (doc and len(doc)) else "X"
                cat = "function" if pos in FUNC else ("content" if pos in CONTENT else "other")
            disagree = (pf[:, t] != pq[:, t]).numpy().astype(float)   # [L]
            cats.setdefault(cat, {"sum": np.zeros(L), "n": 0})
            cats[cat]["sum"] += disagree
            cats[cat]["n"] += 1
            if marg is not None:
                (margin_flip if pf[E_probe, t] != pq[E_probe, t] else margin_same).append(float(marg[E_probe, t]))

    # ---- print D(E) by category (early E<=8 mean) ----
    print(f"\n==================== D(E) by token category — {args.tag} (4bit vs fp16) ====================")
    print(f"{'category':12} {'n':>7} {'D(E<=8)':>9} {'D(E>=16)':>9} {'D(all)':>8}")
    rows = []
    for cat, v in sorted(cats.items(), key=lambda x: -x[1]["n"]):
        if v["n"] == 0:
            continue
        D = v["sum"] / v["n"]                       # [L]
        rows.append((cat, v["n"], D[:8].mean(), D[15:].mean(), D[:L-1].mean()))
    for cat, n, de, dl, da in rows:
        print(f"{cat:12} {n:>7} {de:>9.3f} {dl:>9.3f} {da:>8.3f}")

    # ---- margin <-> D correlation ----
    if has_margin and margin_flip and margin_same:
        mf, ms = np.mean(margin_flip), np.mean(margin_same)
        print(f"\n-- margin<->flip at layer E={E_probe+1} (fp16 top1-top2 logit gap) --")
        print(f"   flipped-by-4bit positions: mean margin {mf:.3f} (n={len(margin_flip)})")
        print(f"   unchanged positions:       mean margin {ms:.3f} (n={len(margin_same)})")
        print(f"   -> flips happen at {ms/mf:.1f}x SMALLER margin (close calls flip)" if mf > 0 else "")

    out = {"tag": args.tag, "kind": args.kind,
           "D_by_category": {c: {"n": int(n), "D_early": float(de), "D_late": float(dl), "D_all": float(da)}
                             for c, n, de, dl, da in rows}}
    if has_margin and margin_flip and margin_same:
        out["margin_at_E"] = E_probe + 1
        out["margin_flipped_mean"] = float(np.mean(margin_flip))
        out["margin_unchanged_mean"] = float(np.mean(margin_same))
    json.dump(out, open(os.path.join(args.shadow_dir, "tokencat_summary.json"), "w"), indent=2)
    print(f"[save] -> {args.shadow_dir}/tokencat_summary.json")


if __name__ == "__main__":
    main()
