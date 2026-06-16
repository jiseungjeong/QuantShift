"""MAG (Margin-Adaptive gamma) preliminary probe (offline).

Idea: in the draft phase, watch each draft token's top1-top2 margin; if margin < tau_m, STOP the draft
block at that position (don't propose the rest). Cutting near-boundary draft tokens (where 4bit flips
concentrate, flip-margin 9.3x smaller) saves wasted dequant work -> directly attacks nf4's A<1.
Greedy verify is unchanged -> output is bit-identical, only speed changes (consistent with Delta-alpha~=0).

This is an OFFLINE simulation on the saved shadow preds+margin (no GPU). It reconstructs, for a fixed
exit layer E (headline E=8 = LayerSkip default), the per-position draft/verify match AND the nf4 draft's
own margin (margin[E, t]) -- then simulates "cut block when draft margin < tau_m" and recomputes both:
  - l_bar(tau_m): mean tokens advanced per verify cycle (acceptance side -- shrinks as we cut)
  - mean draft steps per cycle (cost side -- shrinks as we cut)

COST MODEL (clearly labeled; A is an ESTIMATE, like the gamma-axis in report_extra.py):
  Per verify-cycle throughput  T = l_bar / (draft_steps * r + 1),  r = c_draft/c_verify.
  We report A RELATIVE to the no-cut sim baseline so absolute shadow-vs-measured scale cancels:
      A_est(tau) = A_measured * [ T_sim(tau) / T_sim(no-cut) ].
  By construction A_est(tau=0) == A_measured (exact anchor to breakeven.json). r is back-solved so the
  no-cut sim operating point reproduces the measured A: r = (l_bar0/A_meas - 1)/draft0, using the SIM's
  OWN no-cut l_bar0 & draft0 (not gamma) so the ratio is internally consistent. MAG changes
  (l_bar, draft_steps) -> T_sim(tau) -> A_est. The flip A<1 -> A>1 is what the gate tests.

PRE-REGISTERED GATE (locked 2026-06-06, before this sweep -- no cherry-pick):
  PROCEED to a real measured probe ONLY IF some tau_m pushes nf4 A from baseline (<1) to A > 1.0.
  Otherwise -> demote to future-work text (zero loss). Scope: nf4 x {human_eval, cnn_dm}.

IMPORTANT CAVEATS (printed in output, mirrored in plan.md):
  (1) draft margin here is the nf4 EARLY-EXIT margin (margin[E,t]) -- the correct MAG signal, so this is
      a near-measurement sim; a real run only adds the dynamic-stop overhead (small).
  (2) A is a COST-MODEL ESTIMATE (r back-solved); the l_bar<->draft_steps trade-off is real but the exact
      wall-clock needs the measured probe. Do not report A as measured.
  (3) cutting draft also lowers l_bar -- the "A<1 -> A>1" flip is NOT guaranteed; that's what we test.
"""
import json, os, statistics
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "logs", "report")
os.makedirs(OUT, exist_ok=True)
GAMMA = 6
E_INDEX = 7          # exit layer E=8 (0-based index 7) -- LayerSkip default / headline
TAUS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]   # tau_m sweep (logit-gap units)

TASKS = {
    "human_eval": os.path.join(ROOT, "logs", "shadow"),
    "cnn_dm":     os.path.join(ROOT, "logs", "shadow", "cnn_dm"),
}


def gen_region(s):
    plen, S = s["prompt_len"], len(s["ids"])
    lo, hi = plen - 1, S - 1
    return (lo, hi) if hi > lo else None


def mag_block_sim(match, margin, gamma, tau):
    """Simulate MAG block acceptance + draft-step cost over one sequence's generated region.

    Standard self-SD block (accept-until-first-reject) but the draft phase STOPS early at the first
    position whose draft margin < tau (that token is the last draft proposed in the block).

    Returns (total_accepted, total_proposed, total_draft_steps, n_cycles) accumulated over the sequence.
      - accepted/proposed -> acceptance; advanced tokens = accepted + (1 verify bonus per cycle)
      - draft_steps = number of draft forwards actually run (the cost MAG reduces)
    Token t in a block is "proposed"; we run its draft forward (1 draft step) THEN decide:
      * if it matches verify -> accepted, continue to next draft token (unless we already hit a cut)
      * if it mismatches    -> rejected, block ends (verify supplies the corrected token)
      * a draft token with margin < tau is the LAST proposed in the block (cut AFTER proposing it):
        we still ran its forward (counts as a draft step) and it can still be accepted/rejected,
        but we do not propose further draft tokens this block.
    """
    m = [bool(x) for x in match]
    mg = list(margin)
    n = len(m)
    pos = 0
    tot_acc = tot_prop = tot_draft = n_cycle = 0
    while pos < n:
        block = list(range(pos, min(pos + gamma, n)))
        acc = 0
        draft_steps = 0
        ended = False
        for j, t in enumerate(block):
            draft_steps += 1          # we ran this draft token's forward
            if m[t]:
                acc += 1
            else:
                ended = True          # mismatch -> reject, block ends, verify corrects
                break
            if mg[t] < tau:           # MAG cut: this is the last draft token proposed this block
                ended = True
                break
        proposed = draft_steps        # tokens we actually proposed (= draft forwards)
        tot_prop += proposed
        tot_acc += acc
        tot_draft += draft_steps
        n_cycle += 1
        # advance: accepted draft tokens + 1 verify bonus token (always, on full/partial/cut block)
        advance = acc + 1
        pos += max(advance, 1)
    return tot_acc, tot_prop, tot_draft, n_cycle


def sim_one_tau(p, fp, tau):
    """Walk all sequences at one tau -> aggregate (l_bar_sim, mean_draft_steps, acceptance)."""
    acc_t = prop_t = draft_t = cyc_t = 0
    for sq, sf in zip(p, fp):
        if sq["ids"] != sf["ids"]:
            continue
        reg = gen_region(sq)
        if not reg:
            continue
        lo, hi = reg
        preds = sq["preds"].long()
        match = (preds[E_INDEX, lo:hi] == preds[-1, lo:hi]).tolist()   # draft(E)==verify(L), nf4
        mg = sq["margin"][E_INDEX, lo:hi].float().tolist()            # nf4 draft margin at E
        a, prop, dr, ncy = mag_block_sim(match, mg, GAMMA, tau)
        if ncy == 0:
            continue
        acc_t += a; prop_t += prop; draft_t += dr; cyc_t += ncy
    if cyc_t == 0:
        return None
    l_bar = (acc_t + cyc_t) / cyc_t          # advanced tokens (accepted + 1 verify bonus) per cycle
    mean_draft = draft_t / cyc_t             # draft forwards per cycle (the cost MAG reduces)
    acceptance = acc_t / prop_t if prop_t else float("nan")
    return {"l_bar": l_bar, "mean_draft": mean_draft, "acceptance": acceptance}


def probe_task(task, sh_dir, A_meas, l_bar_meas):
    p = torch.load(os.path.join(sh_dir, "preds_nf4.pt"))
    fp = torch.load(os.path.join(sh_dir, "preds_fp16.pt"))
    if "margin" not in p[0]:
        return {"error": f"preds_nf4.pt in {sh_dir} has no margin -- rerun shadow_landscape --precision nf4 --save-margin"}

    # --- no-cut sim baseline: cut triggers when margin<tau, so tau<=0 never cuts (all margins>=0) ---
    base = sim_one_tau(p, fp, tau=-1.0)
    if base is None:
        return {"error": f"{task}: no usable cycles"}
    l0, d0 = base["l_bar"], base["mean_draft"]
    # back-solve r so the no-cut sim point reproduces measured A: A_meas = l0/(d0*r+1) -> r=(l0/A_meas-1)/d0
    r = (l0 / A_meas - 1.0) / d0
    T0 = l0 / (d0 * r + 1.0)                  # == A_meas by construction (anchor)

    rows = []
    for tau in TAUS:
        s = sim_one_tau(p, fp, tau)
        if s is None:
            continue
        T = s["l_bar"] / (s["mean_draft"] * r + 1.0)
        A_est = A_meas * (T / T0)             # relative to no-cut -> tau=0 reproduces ~A_meas
        rows.append({"tau": tau, "acceptance": round(s["acceptance"], 4),
                     "l_bar": round(s["l_bar"], 3), "mean_draft_steps": round(s["mean_draft"], 3),
                     "A_est": round(A_est, 4)})
    return {"r_costratio": round(r, 4), "A_meas_baseline": A_meas,
            "l_bar_sim_nocut": round(l0, 3), "draft_sim_nocut": round(d0, 3), "sweep": rows}


def main():
    bj = json.load(open(os.path.join(ROOT, "logs", "baselines", "breakeven.json")))
    results = {}
    for task, sh in TASKS.items():
        nf = bj.get(task, {}).get("nf4")
        if not nf:
            print(f"[skip] {task}: no nf4 in breakeven.json"); continue
        res = probe_task(task, sh, nf["A"], nf["l_bar"])
        results[task] = res
        print(f"\n==================== MAG PROBE (offline, cost-model) — {task} (nf4, E=8) ====================")
        if "error" in res:
            print("  [error]", res["error"]); continue
        print(f"  anchor: measured A={res['A_meas_baseline']} | sim no-cut l_bar={res['l_bar_sim_nocut']} "
              f"draft={res['draft_sim_nocut']} -> back-solved r=c_draft/c_verify={res['r_costratio']}")
        print(f"  {'tau_m':>6} {'accept':>7} {'l_bar':>7} {'draft_steps':>11} {'A_est':>7}  note")
        best = None
        for row in res["sweep"]:
            note = ""
            if row["tau"] == 0.0:
                note = "(no cut = baseline check)"
            if row["A_est"] > 1.0 and (best is None or row["A_est"] > best["A_est"]):
                best = row
            print(f"  {row['tau']:>6.2f} {row['acceptance']:>7.3f} {row['l_bar']:>7.3f} "
                  f"{row['mean_draft_steps']:>11.3f} {row['A_est']:>7.3f}  {note}")
        if best:
            print(f"  >>> GATE PASS: tau_m={best['tau']} gives A_est={best['A_est']} > 1.0 "
                  f"(baseline {res['A_meas_baseline']}) -> proceed to measured probe candidate")
        else:
            amax = max((r["A_est"] for r in res["sweep"]), default=float("nan"))
            print(f"  >>> GATE FAIL: max A_est={amax:.3f} <= 1.0 -> demote to future-work (zero loss)")

    json.dump(results, open(os.path.join(OUT, "mag_probe.json"), "w"), indent=2)

    # ---- plot: A_est vs tau_m, both tasks, with A=1 gate line ----
    plt.figure(figsize=(6.4, 4.3))
    for task, res in results.items():
        if "error" in res or not res.get("sweep"):
            continue
        xs = [r["tau"] for r in res["sweep"]]
        ys = [r["A_est"] for r in res["sweep"]]
        plt.plot(xs, ys, "o-", label=f"{task} (nf4, baseline A={res['A_meas_baseline']})")
    plt.axhline(1.0, color="k", ls="--", lw=1, label="A=1 gate (q-self-SD = q-AR)")
    plt.xlabel("MAG draft-cut threshold tau_m (top1-top2 logit gap)")
    plt.ylabel("A_est = q-self-SD / q-AR  (cost-model ESTIMATE)")
    plt.title("MAG probe (offline): does cutting near-boundary draft push nf4 A>1?")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "mag_probe.png"), dpi=130); plt.close()
    print(f"\n[caveat] A_est is a COST-MODEL ESTIMATE (r back-solved from measured A); "
          f"draft margin = nf4 early-exit margin[E=8] (correct MAG signal, near-measurement sim).")
    print(f"[save] -> {OUT}/(mag_probe.json, mag_probe.png)")


if __name__ == "__main__":
    main()
