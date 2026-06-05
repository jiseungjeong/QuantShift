"""QuantShift §4/§7.2: aggregate baseline JSONs into the B1-B6 table + speedup + latency story.

Reads logs/baselines/baseline_{fp16,nf4,int8}.json (from baseline_eval.py) and prints:
  - B1-B6 table: tokens/s, speedup vs B1 (same dataset), quality, VRAM, disk, load time
  - latency breakdown WITHOUT a profiler: delta(time/token) between precisions at matched (strategy,dataset)
    = the de-quantization tax per token (same FLOPs, only difference is dequant/unpack).

Run: conda run -n NLP env PYTHONNOUSERSITE=1 python aggregate_baselines.py
"""
import json, os

BDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "baselines")
PRECS = ["fp16", "int8", "nf4"]                 # bf16->8bit->4bit order
# B-id map: (strategy, precision) -> label
BID = {("ar", "fp16"): "B1", ("selfsd", "fp16"): "B2",
       ("ar", "nf4"): "B3", ("selfsd", "nf4"): "B4",
       ("ar", "int8"): "B5", ("selfsd", "int8"): "B6"}


def load():
    d = {}
    for p in PRECS:
        path = os.path.join(BDIR, f"baseline_{p}.json")
        if os.path.exists(path):
            d[p] = json.load(open(path))
            # override buggy inline pass@1 with the rescored value (indent-fixed).
            # self-SD greedy output == AR output at same precision -> same pass@1 for both strategies.
            pk = os.path.join(BDIR, f"passk_{p}.json")
            if os.path.exists(pk):
                rate = json.load(open(pk))["pass@1"]
                for strat in ("ar", "selfsd"):
                    c = d[p]["configs"].get(f"{strat}/human_eval")
                    if c:
                        c["quality"] = {"pass@1": rate}
    return d


def qstr(q):
    if "pass@1" in q:
        return f"pass@1={q['pass@1']:.3f}"
    if "rouge2" in q:
        return f"rouge2={q['rouge2']:.3f}"
    return str(q)


def main():
    d = load()
    if not d:
        print(f"[err] no baseline_*.json in {BDIR}"); return

    for dataset in ["human_eval", "cnn_dm"]:
        print(f"\n==================== §4 BASELINES — {dataset} ====================")
        # B1 (AR fp16) tokens/s as 1.00x reference for this dataset
        ref = d.get("fp16", {}).get("configs", {}).get(f"ar/{dataset}", {}).get("tokens_per_sec_mean")
        print(f"{'B':3} {'config':16} {'tok/s':>10} {'speedup':>8} {'quality':>16} {'acc':>6} {'VRAM':>6} {'disk':>6}")
        for prec in PRECS:
            if prec not in d:
                continue
            disk = d[prec]["disk_gb"]
            for strat in ["ar", "selfsd"]:
                c = d[prec]["configs"].get(f"{strat}/{dataset}")
                if not c:
                    continue
                bid = BID[(strat, prec)]
                tps = c["tokens_per_sec_mean"]
                sp = (tps / ref) if (ref and tps) else float("nan")
                acc = c["acceptance_mean"]
                accs = f"{acc:.3f}" if acc is not None else "  -  "
                cfg = f"{strat},{prec}"
                print(f"{bid:3} {cfg:16} {tps:>10.1f} {sp:>7.2f}x {qstr(c['quality']):>16} {accs:>6} {c['vram_peak_gb']:>5.1f}G {disk:>5.1f}G")

        # ---- latency breakdown via precision delta (no profiler) ----
        print(f"  -- de-quant tax (time/token delta vs fp16, matched strategy) --")
        for strat in ["ar", "selfsd"]:
            base = d.get("fp16", {}).get("configs", {}).get(f"{strat}/{dataset}", {}).get("time_per_token_mean")
            if base is None:
                continue
            for prec in ["int8", "nf4"]:
                if prec not in d:
                    continue
                tpt = d[prec]["configs"].get(f"{strat}/{dataset}", {}).get("time_per_token_mean")
                if tpt is None:
                    continue
                dtax = (tpt - base) * 1000  # ms/token
                ratio = tpt / base
                print(f"     {strat:7} {prec}: {tpt*1000:.1f} ms/tok vs fp16 {base*1000:.1f}  -> +{dtax:.1f} ms ({ratio:.2f}x)  [= dequant overhead]")

    print("\n[note] speedup = tokens/s vs B1(AR,fp16) per dataset. de-quant tax = extra time/token of quantized vs fp16 (same compute, only dequant differs).")


if __name__ == "__main__":
    main()
