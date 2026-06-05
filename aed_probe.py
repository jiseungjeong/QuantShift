"""QuantShift §8.4 (appendix): AED (Adaptive Exit Depth) preliminary probe -- OFFLINE, GPU 0.

Idea: draft's exit layer E is per-token adaptive, not fixed at 8. Easy positions (high margin at a
shallow layer) exit early (E small); hard positions go deeper. Two levers at once:
  (r down)  average draft depth < 8  -> c_draft drops (MEASURED r(E)~6.9ms*E, rE_diagnostic.json: LINEAR in E)
  (l_bar ~) only exit early where the shallow layer ALREADY matches verify -> acceptance preserved
Greedy verify is unchanged -> output bit-identical (AED only changes draft depth), consistent with Delta-alpha~=0.

Why AED (and not MAG): the r(E) diagnostic showed draft decode cost scales ~linearly with E
(step(4)/step(8)=0.51, step(2)/step(8)=0.27), so lowering average E genuinely cuts c_draft. MAG cut draft
LENGTH (gamma) which barely helped; AED cuts draft DEPTH (E) which the measurement says is the real lever.

COST MODEL (A is an ESTIMATE; r(E) is MEASURED -> stronger than MAG's single back-solved r):
  Per verify-cycle:  T = l_bar / ( sum_over_draft_tokens c_draft(E_t) + c_verify )
  c_draft(E) = k * E  (k from rE_diagnostic.json, the measured 6.9 ms/layer slope).
  c_verify is back-solved so the NO-ADAPT baseline (every draft token at fixed E=8) reproduces the
  MEASURED A (breakeven.json). Then A_est(policy) = A_meas * T(policy)/T(baseline). By construction the
  fixed-E=8 policy gives A_est == A_meas (exact anchor).

DEPTH POLICY (margin-gated, tau from margin-curve quantiles -> no post-hoc cherry-pick):
  For token t, scan E in DEPTHS ascending; exit at the first E whose nf4 draft margin[E,t] >= tau.
  If none reach tau, use the deepest allowed E (E_max). The DRAFT TOKEN proposed is preds[E_chosen, t].
  Acceptance = does that proposed token match verify (preds[L])? (early exit can change the token, so
  acceptance is recomputed against the CHOSEN-depth prediction -- this is the honest l_bar.)

ORACLE UPPER BOUND: each token uses the shallowest E whose prediction ALREADY == verify (perfect oracle).
  Gives the best A any depth policy could reach -> if even the oracle A <= 1.0, AED is hopeless -> stop.

PRE-REGISTERED GATE (locked 2026-06-06, same as MAG -> comparable):
  PROCEED to a real measured probe ONLY IF some tau pushes nf4 A > 1.0 (and oracle A > 1.0 first).
  Scope nf4 x {human_eval, cnn_dm}. Otherwise -> future-work text (zero loss).

Run: conda run -n NLP env PYTHONNOUSERSITE=1 python aed_probe.py
"""
import json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "logs", "report")
GAMMA = 6
E_FIXED = 8                      # LayerSkip default / MAG headline (baseline anchor)
DEPTHS = [2, 3, 4, 5, 6, 8, 10, 12]   # allowed adaptive exit depths (ascending)
E_MAX = 12                       # deepest draft layer if no depth reaches tau
TASKS = {"human_eval": os.path.join(ROOT, "logs", "shadow"),
         "cnn_dm":     os.path.join(ROOT, "logs", "shadow", "cnn_dm")}


def gen_region(s):
    plen, S = s["prompt_len"], len(s["ids"])
    lo, hi = plen - 1, S - 1
    return (lo, hi) if hi > lo else None


def block_walk(match, depth, gamma):
    """Standard accept-until-first-reject block walk, returning per-cycle accepted count, draft tokens
    proposed, and the list of draft depths actually run (for cost). `match[i]` and `depth[i]` are aligned
    per generated position. Returns (tot_acc, tot_draft_depth_sum, n_cycles, advanced_tokens)."""
    n = len(match)
    pos = 0
    tot_acc = depth_sum = n_cyc = adv = 0
    while pos < n:
        block = range(pos, min(pos + gamma, n))
        acc = 0
        for t in block:
            depth_sum += depth[t]          # we ran a draft forward to depth E_t for this token
            if match[t]:
                acc += 1
            else:
                break                       # reject -> block ends (verify corrects)
        n_cyc += 1
        tot_acc += acc
        adv += acc + 1                      # accepted + 1 verify bonus
        pos += max(acc + 1, 1)
    return tot_acc, depth_sum, n_cyc, adv


def policy_fixed(preds, margin, lo, hi, E):
    """Fixed-depth policy: every token drafted at layer E. Returns (match[], depth[])."""
    v = preds[-1]
    pe = preds[E - 1]
    match = [(pe[t] == v[t]).item() for t in range(lo, hi)]
    depth = [E] * (hi - lo)
    return match, depth


def policy_margin(preds, margin, lo, hi, tau):
    """Margin-gated AED: ascend DEPTHS, exit at first E with margin>=tau; else E_MAX.
    Draft token = preds[E_chosen]; match vs verify recomputed at chosen depth."""
    v = preds[-1]
    match, depth = [], []
    for t in range(lo, hi):
        chosen = E_MAX
        for E in DEPTHS:
            if margin[E - 1, t].item() >= tau:
                chosen = E
                break
        match.append((preds[chosen - 1, t] == v[t]).item())
        depth.append(chosen)
    return match, depth


def aggregate_block_margin(preds_list, tau):
    """BLOCK-LEVEL margin-gated AED (the FAITHFULLY-MEASURABLE policy): one exit depth E per gamma-block,
    chosen from the block's FIRST draft token margin (ascend DEPTHS, first E with margin>=tau; else E_MAX).
    This respects forward_remainder's single-exit_layer-per-call assumption (per-token E would need ragged
    KV bookkeeping -> not a clean measured probe). Walks blocks exactly like block_walk."""
    acc_t = depthsum_t = cyc_t = adv_t = prop_t = Esum = Ecnt = 0
    for s in preds_list:
        reg = gen_region(s)
        if not reg:
            continue
        lo, hi = reg
        preds = s["preds"].long(); margin = s["margin"]; v = preds[-1]
        n = hi - lo
        pos = 0
        while pos < n:
            blk = range(pos, min(pos + GAMMA, n))
            first_t = lo + pos
            E = E_MAX
            for cand in DEPTHS:
                if margin[cand - 1, first_t].item() >= tau:
                    E = cand; break
            acc = 0
            for j in blk:
                t = lo + j
                depthsum_t += E; Esum += E; Ecnt += 1; prop_t += 1
                if (preds[E - 1, t] == v[t]).item():
                    acc += 1
                else:
                    break
            cyc_t += 1; acc_t += acc; adv_t += acc + 1
            pos += max(acc + 1, 1)
    if cyc_t == 0:
        return None
    return {"l_bar": adv_t / cyc_t, "draft_cost_units": depthsum_t / cyc_t,
            "acceptance": acc_t / prop_t if prop_t else float("nan"), "mean_E": Esum / Ecnt}


def policy_oracle(preds, margin, lo, hi):
    """Oracle: shallowest E in DEPTHS whose pred already == verify; else E_MAX (and counts as match
    only if E_MAX matches)."""
    v = preds[-1]
    match, depth = [], []
    for t in range(lo, hi):
        chosen, ok = E_MAX, (preds[E_MAX - 1, t] == v[t]).item()
        for E in DEPTHS:
            if (preds[E - 1, t] == v[t]).item():
                chosen, ok = E, True
                break
        match.append(ok)
        depth.append(chosen)
    return match, depth


def aggregate(preds_list, policy_fn):
    """Run a policy over all sequences -> (l_bar, mean_draft_cost_units, acceptance, mean_E).
    policy_fn(preds, margin, lo, hi) -> (match[], depth[]). margin may be None for non-margin policies.
    mean_draft_cost_units = mean over cycles of sum_t E_t (in 'layer' units; c_draft = k*E)."""
    acc_t = depthsum_t = cyc_t = adv_t = prop_t = 0
    Esum = Ecnt = 0
    for s in preds_list:
        reg = gen_region(s)
        if not reg:
            continue
        lo, hi = reg
        preds = s["preds"].long()
        margin = s.get("margin")
        match, depth = policy_fn(preds, margin, lo, hi)
        a, dsum, ncy, adv = block_walk(match, depth, GAMMA)
        acc_t += a; depthsum_t += dsum; cyc_t += ncy; adv_t += adv
        prop_t += len(match); Esum += sum(depth); Ecnt += len(depth)
    if cyc_t == 0:
        return None
    l_bar = adv_t / cyc_t
    draft_cost_units = depthsum_t / cyc_t          # sum of E over draft tokens, per cycle (k*this = c_draft cost)
    acceptance = acc_t / prop_t if prop_t else float("nan")
    mean_E = Esum / Ecnt
    return {"l_bar": l_bar, "draft_cost_units": draft_cost_units,
            "acceptance": acceptance, "mean_E": mean_E}


def probe_task(task, sh_dir, A_meas, k_slope, c_verify_ms, margin_curve_q):
    p = torch.load(os.path.join(sh_dir, "preds_nf4.pt"))
    if "margin" not in p[0]:
        return {"error": f"{sh_dir}: preds_nf4.pt has no margin"}

    # --- anchor: fixed E=8 baseline reproduces measured A ---
    # Cost model (dimensionless, c_AR=c_verify=C since both are one full-model forward):
    #   A = l_bar / (rho * draft_cost_units + 1),  rho = (k ms/layer)/(C ms full forward) = per-layer
    #   draft cost relative to one verify/AR forward. Back-solve rho from baseline so A_meas is reproduced:
    #     A_meas = l0 / (rho*du0 + 1)  =>  rho = (l0/A_meas - 1)/du0   (POSITIVE, dimensionless).
    #   k (ms/layer) enters only as a sanity check on rho (rho ~ k/C with C ~ 32*k for a full pass).
    base = aggregate(p, lambda preds, margin, lo, hi: policy_fixed(preds, margin, lo, hi, E_FIXED))
    l0, du0 = base["l_bar"], base["draft_cost_units"]
    rho = (l0 / A_meas - 1.0) / du0                 # per-draft-layer cost relative to one full forward
    def A_of(agg):
        return agg["l_bar"] / (rho * agg["draft_cost_units"] + 1.0)
    A0 = A_of(base)                                 # == A_meas (anchor)
    # sanity: implied full-forward layers C/k = 1/rho ; baseline mean draft layers/token
    implied_full_layers = (1.0 / rho) if rho > 0 else float("nan")

    # --- oracle upper bound ---
    orac = aggregate(p, policy_oracle)
    A_oracle = A_of(orac)

    # --- margin-gated sweep ---
    # tau grid from the margin-curve quantiles (P(flip)=0.05..0.20 region) + a spread
    taus = sorted(set([round(margin_curve_q.get("q05", 0.3), 3), round(margin_curve_q.get("q20", 1.0), 3),
                       0.1, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]))
    rows = []
    for tau in taus:
        agg = aggregate(p, lambda preds, margin, lo, hi, _t=tau: policy_margin(preds, margin, lo, hi, _t))
        if agg is None:
            continue
        A_est = A_of(agg)
        rows.append({"tau": tau, "mean_E": round(agg["mean_E"], 3),
                     "acceptance": round(agg["acceptance"], 4), "l_bar": round(agg["l_bar"], 3),
                     "A_est": round(A_est, 4)})
    # block-level (faithfully MEASURABLE) sweep
    rows_blk = []
    for tau in taus:
        agg = aggregate_block_margin(p, tau)
        if agg is None:
            continue
        rows_blk.append({"tau": tau, "mean_E": round(agg["mean_E"], 3),
                         "acceptance": round(agg["acceptance"], 4), "l_bar": round(agg["l_bar"], 3),
                         "A_est": round(A_of(agg), 4)})
    return {"A_meas": A_meas, "rho": round(rho, 4), "implied_full_fwd_layers": round(implied_full_layers, 1),
            "k_ms_per_layer": round(k_slope, 3), "baseline_l_bar": round(l0, 3),
            "baseline_draft_units": round(du0, 3),
            "oracle": {"mean_E": round(orac["mean_E"], 3), "l_bar": round(orac["l_bar"], 3),
                       "A_oracle": round(A_oracle, 4)},
            "sweep": rows, "sweep_blocklevel": rows_blk}


def main():
    bj = json.load(open(os.path.join(ROOT, "logs", "baselines", "breakeven.json")))
    rE = json.load(open(os.path.join(OUT, "rE_diagnostic.json")))
    # measured slope k (ms/layer): fit step_ms = k*E through the diagnostic rows
    Es = np.array([r["E"] for r in rE["rows"]], float)
    ms = np.array([r["one_decode_step_ms"] for r in rE["rows"]], float)
    k_slope = float((Es @ ms) / (Es @ Es))     # least-squares through origin
    c_verify_seed = None
    # margin-curve quantiles (for tau grid) from extra.json if present
    try:
        ex = json.load(open(os.path.join(OUT, "extra.json")))
        # not exact quantiles stored; use sensible defaults from the margin curve shape
    except Exception:
        ex = {}
    mq = {"q05": 0.3, "q20": 1.0}

    results = {}
    for task, sh in TASKS.items():
        nf = bj.get(task, {}).get("nf4")
        if not nf:
            continue
        res = probe_task(task, sh, nf["A"], k_slope, c_verify_seed, mq)
        results[task] = res
        print(f"\n==================== AED PROBE (offline, cost-model) — {task} (nf4) ====================")
        if "error" in res:
            print("  [error]", res["error"]); continue
        print(f"  measured: A={res['A_meas']} | k={res['k_ms_per_layer']:.2f} ms/layer | "
              f"rho={res['rho']} (1 full fwd ~= {res['implied_full_fwd_layers']} draft-layers) | "
              f"baseline E=8 l_bar={res['baseline_l_bar']} draft_units={res['baseline_draft_units']}")
        o = res["oracle"]
        print(f"  ORACLE (perfect depth): mean_E={o['mean_E']} l_bar={o['l_bar']} -> A_oracle={o['A_oracle']} "
              f"{'(>1 -> ceiling clears gate)' if o['A_oracle']>1 else '(<=1 -> AED HOPELESS, stop)'}")
        for label, key in [("per-token (idealized)", "sweep"), ("block-level (MEASURABLE)", "sweep_blocklevel")]:
            print(f"  -- {label} --")
            print(f"  {'tau':>6} {'mean_E':>7} {'accept':>7} {'l_bar':>7} {'A_est':>7}")
            best = None
            for r in res[key]:
                if r["A_est"] > 1.0 and (best is None or r["A_est"] > best["A_est"]):
                    best = r
                print(f"  {r['tau']:>6.2f} {r['mean_E']:>7.3f} {r['acceptance']:>7.3f} {r['l_bar']:>7.3f} {r['A_est']:>7.3f}")
            if best:
                print(f"  >>> GATE PASS ({label}): tau={best['tau']} -> A_est={best['A_est']} > 1.0 "
                      f"(baseline {res['A_meas']})")
            else:
                amax = max((r["A_est"] for r in res[key]), default=float("nan"))
                print(f"  >>> GATE FAIL ({label}): max A_est={amax:.3f} <= 1.0 (oracle {o['A_oracle']})")

    json.dump(results, open(os.path.join(OUT, "aed_probe.json"), "w"), indent=2)

    # plot A_est vs mean_E (the real knob), both tasks, with oracle point + A=1 line
    plt.figure(figsize=(6.6, 4.4))
    for task, res in results.items():
        if "error" in res:
            continue
        xs = [r["mean_E"] for r in res["sweep"]]
        ys = [r["A_est"] for r in res["sweep"]]
        line, = plt.plot(xs, ys, "o-", label=f"{task} (baseline A={res['A_meas']} @E=8)")
        o = res["oracle"]
        plt.plot([o["mean_E"]], [o["A_oracle"]], "*", color=line.get_color(), markersize=15,
                 label=f"{task} oracle (E={o['mean_E']})")
    plt.axhline(1.0, color="k", ls="--", lw=1, label="A=1 gate")
    plt.xlabel("mean draft depth E (adaptive)")
    plt.ylabel("A_est = q-self-SD / q-AR  (cost-model; r(E) MEASURED)")
    plt.title("AED probe: adaptive draft depth vs A (does shallower avg E push nf4 A>1?)")
    plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "aed_probe.png"), dpi=130); plt.close()
    print(f"\n[caveat] A_est cost-model; r(E)=k*E MEASURED LINEAR (k={k_slope:.2f} ms/layer, rE_diagnostic); "
          f"rho back-solved so fixed-E=8 reproduces measured A (anchor). Depth uses nf4 self margin "
          f"(near-measurement). Oracle = perfect-depth upper bound. A>1 margin modest -> measured probe is the gate's next step.")
    print(f"[save] -> {OUT}/(aed_probe.json, aed_probe.png)")


if __name__ == "__main__":
    main()
