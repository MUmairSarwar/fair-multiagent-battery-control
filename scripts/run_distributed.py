"""Distributed optimisation experiments.

    python scripts/run_distributed.py            # full run (about 30-40 minutes on 2 cores)
    python scripts/run_distributed.py --quick    # a few days and iterations, for CI

Writes results/distributed/*.csv, *.md, summary.json and figures/*.png.

1. Main comparison on the 51 test days (same loads as the RL results):
   centralised LP, exact MILP, dual decomposition and ADMM.
2. Fairness: efficient vs proportional vs max-min sharing of the transformer
   (hard limit), centralised and with fair ADMM.
3. Scaling: 6, 24 and 96 synthetic households on a few days.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fairbattery.baselines import solve_day_lp  # noqa: E402
from fairbattery.distributed import (  # noqa: E402
    dual_decomposition,
    from_env,
    scaled_problem,
    sharing_admm,
    solve_centralized,
)
from fairbattery.experiment import load_config, make_env, test_loads  # noqa: E402
from fairbattery.metrics import jain_index  # noqa: E402

OUT = ROOT / "results" / "distributed"
FIG = OUT / "figures"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8986"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})

# Settings chosen with a small sweep on two test days (step0 in 0.005..0.1, rho in 0.5..5).
DUAL_STEP0 = 0.02
RHO = 0.5

ENV = None
TEST_DAYS = None
ITERS = 300


def _setup():
    global ENV, TEST_DAYS
    cfg = load_config()
    env, _, test_days = make_env(cfg)
    loads = test_loads(env, test_days, cfg["eval"]["load_seed"])
    env.reset(test_days, np.random.default_rng(0), loads=loads)
    ENV, TEST_DAYS = env, test_days


def _first_below(gap, thr=0.01):
    gap = np.asarray(gap)
    hit = np.where(gap < thr)[0]
    return int(hit[0]) + 1 if len(hit) else None


def main_job(b):
    warnings.filterwarnings("ignore")
    p = from_env(ENV, b)
    c, d = solve_day_lp(ENV, b)
    lp = p.system_cost(c, d)
    simult = bool(((c > 1e-6) & (d > 1e-6)).any())
    ce, de = solve_day_lp(ENV, b, exclusive=True) if simult else (c, d)
    exact = p.system_cost(ce, de)

    t0 = time.time()
    dual = dual_decomposition(p, iters=ITERS, step0=DUAL_STEP0, reference_cost=lp)
    t_dual = time.time() - t0
    t0 = time.time()
    adm = sharing_admm(p, rho=RHO, iters=ITERS)
    t_admm = time.time() - t0
    admm_gap = np.abs(np.array(adm.history["system_cost"]) - lp) / max(abs(lp), 1e-9)
    return {
        "day": int(b),
        "lp_cost": lp,
        "milp_cost": exact,
        "lp_has_simultaneous": simult,
        "dual_cost": dual.history["primal_cost_avg"][-1],
        "dual_bound": dual.history["dual_bound"][-1],
        "dual_overload_kwh": dual.history["overload_avg"][-1],
        "dual_iters_to_1pct": _first_below(dual.history["gap_rel"]),
        "dual_seconds": t_dual,
        "admm_cost": float(p.system_cost(adm.c, adm.d)),
        "admm_overload_kwh": float(p.overload(adm.c - adm.d).sum()),
        "admm_iters": adm.iters,
        "admm_iters_to_1pct": _first_below(admm_gap),
        "admm_seconds": t_admm,
        "_hist": {"dual_gap": dual.history["gap_rel"], "admm_gap": admm_gap.tolist(),
                  "admm_primal_res": adm.history["primal_res"]},
    }


def fair_job(b):
    warnings.filterwarnings("ignore")
    p = from_env(ENV, b)
    out = {"day": int(b)}
    for obj in ["efficient", "proportional", "maxmin"]:
        c, d = solve_centralized(p, obj)
        out[obj] = (-(c @ p.a + d @ p.b)).tolist()
    adm = sharing_admm(p, rho=RHO, iters=ITERS, alpha=1, hard=True)
    out["admm_proportional"] = (-(adm.c @ p.a + adm.d @ p.b)).tolist()
    out["admm_proportional_iters"] = adm.iters
    out["admm_proportional_violation_kwh"] = adm.history["max_violation_kwh"][-1]
    return out


def scale_job(arg):
    warnings.filterwarnings("ignore")
    n, day = arg
    p = scaled_problem(ENV, day, n, seed=day)
    c, d = solve_centralized(p, "efficient_soft")
    ref = p.system_cost(c, d)
    t0 = time.time()
    adm = sharing_admm(p, rho=RHO, iters=ITERS)
    t_admm = time.time() - t0
    gap = np.abs(np.array(adm.history["system_cost"]) - ref) / max(abs(ref), 1e-9)
    t0 = time.time()
    dual = dual_decomposition(p, iters=ITERS, step0=DUAL_STEP0, reference_cost=ref)
    t_dual = time.time() - t0
    return {
        "households": n, "day": int(day), "reference_cost": ref,
        "admm_iters_to_1pct": _first_below(gap), "admm_final_gap_pct": 100 * gap[-1],
        "admm_seconds_per_household_iter": t_admm / (n * len(gap)),
        "dual_iters_to_1pct": _first_below(dual.history["gap_rel"]),
        "dual_final_gap_pct": 100 * dual.history["gap_rel"][-1],
        "dual_seconds_per_household_iter": t_dual / (n * ITERS),
        "admm_overload_kwh": float(p.overload(adm.c - adm.d).sum()),
    }


def fairness_table(rows, cap):
    names = {"efficient": "Efficient (max total saving)", "proportional": "Proportional fairness",
             "maxmin": "Max-min fairness", "admm_proportional": "Proportional fairness, distributed (ADMM)"}
    eff = np.sum([r["efficient"] for r in rows], axis=0)
    out = []
    for k, label in names.items():
        tot = np.sum([r[k] for r in rows], axis=0)  # per household over all days
        u = tot / cap
        out.append({
            "Allocation": label,
            "Total saving (EUR)": round(float(tot.sum()), 2),
            "Price of fairness (%)": round(100 * (1 - tot.sum() / eff.sum()), 2),
            "Jain index": round(jain_index(u), 4),
            "Lowest saving per kWh of battery (EUR)": round(float(u.min()), 3),
            "Highest saving per kWh of battery (EUR)": round(float(u.max()), 3),
        })
    return pd.DataFrame(out)


def figures(main_rows, fair_rows, scale_df, cap, day_for_plot):
    FIG.mkdir(parents=True, exist_ok=True)
    # 1. convergence on one day
    convergence_figure(next(x for x in main_rows if x["day"] == day_for_plot), day_for_plot)

    # 2. saving per kWh of battery by household and allocation
    keys = [("efficient", "Efficient", GREY), ("proportional", "Proportional", BLUE),
            ("maxmin", "Max-min", GREEN), ("admm_proportional", "Proportional (ADMM)", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.2
    x = np.arange(len(cap))
    for j, (k, label, col) in enumerate(keys):
        u = np.sum([rr[k] for rr in fair_rows], axis=0) / cap
        ax.bar(x + (j - 1.5) * w, u, w, color=col, label=label)
    ax.set_xticks(x, [f"H{i + 1}\n{cap[i]:g} kWh" for i in range(len(cap))])
    ax.set_ylabel("Saving per kWh of battery (EUR)")
    ax.set_title("Who gets the savings, over the 51 test days", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fairness.png")
    plt.close(fig)

    # 3. scaling
    if len(scale_df):
        scaling_figure(scale_df)


def convergence_figure(r, day_for_plot):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.semilogy(np.maximum(r["_hist"]["dual_gap"], 1e-6), color=ORANGE, label="Dual decomposition (averaged schedule)")
    ax.semilogy(np.maximum(r["_hist"]["admm_gap"], 1e-6), color=BLUE, label=f"ADMM, rho = {RHO}")
    ax.axhline(0.01, color=GREY, lw=0.8, ls="--")
    ax.text(len(r["_hist"]["dual_gap"]), 0.0065, "1% gap", color=MUTED, ha="right", fontsize=8)
    ax.set_xlabel("Iteration (one message each way per household)")
    ax.set_ylabel("Relative gap to the LP optimum")
    ax.set_title(f"Convergence on one test day (day index {day_for_plot})", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "convergence.png")
    plt.close(fig)


def scaling_figure(scale_df):
    g = scale_df.groupby("households").agg(admm=("admm_final_gap_pct", "median"),
                                           dual=("dual_final_gap_pct", "median"),
                                           admm_it=("admm_iters_to_1pct", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.semilogy(g["households"], g["admm"], "o-", color=BLUE, label="ADMM (when it stops)")
    ax.semilogy(g["households"], g["dual"], "o-", color=ORANGE, label=f"Dual decomposition (after {ITERS} iterations)")
    for _, row in g.iterrows():
        ax.annotate(f"{row['admm_it']:.0f} it. to 1%", (row["households"], row["admm"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=BLUE)
    ax.axhline(1.0, color=GREY, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks(g["households"], [str(v) for v in g["households"]])
    ax.minorticks_off()
    ax.set_xlabel("Households (synthetic, limit scaled with size)")
    ax.set_ylabel("Gap to the optimum (%)")
    ax.set_title("Same algorithms, larger neighbourhoods (median of 2 days)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "scaling.png")
    plt.close(fig)


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def main():
    global ITERS, OUT, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None, help="output folder (default results/distributed)")
    args = ap.parse_args()
    if args.out is not None:
        OUT, FIG = args.out, args.out / "figures"
    _setup()
    days = np.arange(len(TEST_DAYS))
    scale_ns, scale_days = [6, 24, 96], [10, 30]
    if args.quick:
        ITERS = 30
        days = days[:2]
        scale_ns, scale_days = [6, 12], [10]
    OUT.mkdir(parents=True, exist_ok=True)

    with Pool(args.workers) as pool:
        main_rows = pool.map(main_job, days)
        fair_rows = pool.map(fair_job, days)
        scale_rows = pool.map(scale_job, [(n, d) for n in scale_ns for d in scale_days])

    main_df = pd.DataFrame([{k: v for k, v in r.items() if k != "_hist"} for r in main_rows])
    main_df.to_csv(OUT / "per_day.csv", index=False)
    lp_benefit = -main_df["lp_cost"].sum()
    summary = pd.DataFrame([
        {"Method": "Centralised LP (published bound)", "Net benefit (EUR)": -main_df["lp_cost"].sum(),
         "Share of LP (%)": 100.0, "Overload (kWh)": 0.0, "Iterations (median)": "-"},
        {"Method": "Exact MILP (no simultaneous charge/discharge)", "Net benefit (EUR)": -main_df["milp_cost"].sum(),
         "Share of LP (%)": 100 * -main_df["milp_cost"].sum() / lp_benefit, "Overload (kWh)": 0.0,
         "Iterations (median)": "-"},
        {"Method": f"Dual decomposition ({ITERS} iterations, averaged)",
         "Net benefit (EUR)": -main_df["dual_cost"].sum(),
         "Share of LP (%)": 100 * -main_df["dual_cost"].sum() / lp_benefit,
         "Overload (kWh)": main_df["dual_overload_kwh"].sum(), "Iterations (median)": ITERS},
        {"Method": f"ADMM (rho = {RHO})", "Net benefit (EUR)": -main_df["admm_cost"].sum(),
         "Share of LP (%)": 100 * -main_df["admm_cost"].sum() / lp_benefit,
         "Overload (kWh)": main_df["admm_overload_kwh"].sum(),
         "Iterations (median)": int(main_df["admm_iters"].median())},
    ])
    summary["Net benefit (EUR)"] = summary["Net benefit (EUR)"].round(2)
    summary["Share of LP (%)"] = summary["Share of LP (%)"].round(2)
    summary["Overload (kWh)"] = summary["Overload (kWh)"].round(3)

    cap = ENV.cap
    fair_df = fairness_table(fair_rows, cap)
    scale_df = pd.DataFrame(scale_rows)
    scale_df.to_csv(OUT / "scaling.csv", index=False)
    fair_df.to_csv(OUT / "fairness.csv", index=False)
    summary.to_csv(OUT / "main.csv", index=False)

    scale_summary = scale_df.groupby("households").agg(
        admm_iters_to_1pct=("admm_iters_to_1pct", "median"),
        admm_final_gap_pct_max=("admm_final_gap_pct", "max"),
        dual_iters_to_1pct=("dual_iters_to_1pct", "median"),
        dual_final_gap_pct_max=("dual_final_gap_pct", "max"),
        admm_ms_per_household_iter=("admm_seconds_per_household_iter", lambda s: 1000 * s.mean()),
    ).reset_index().round(3)

    stats = {
        "test_days": int(len(days)),
        "iterations": ITERS,
        "dual_step0": DUAL_STEP0,
        "admm_rho": RHO,
        "days_lp_charges_and_discharges_at_once": int(main_df["lp_has_simultaneous"].sum()),
        "milp_minus_lp_benefit_eur": round(float(-(main_df["milp_cost"].sum() - main_df["lp_cost"].sum())), 4),
        "dual_days_reaching_1pct": int(main_df["dual_iters_to_1pct"].notna().sum()),
        "admm_days_reaching_1pct": int(main_df["admm_iters_to_1pct"].notna().sum()),
        "admm_proportional_max_violation_kwh": float(max(r["admm_proportional_violation_kwh"] for r in fair_rows)),
    }
    for method in ("admm", "dual"):
        reached = main_df[f"{method}_iters_to_1pct"].dropna()
        stats[f"{method}_median_iters_to_1pct"] = float(reached.median()) if len(reached) else None
    summary_json = json.dumps(stats, indent=2, allow_nan=False)
    (OUT / "summary.json").write_text(summary_json)
    (OUT / "tables.md").write_text(
        "## Main comparison\n\n" + md(summary) + "\n\n## Fairness (hard transformer limit)\n\n" + md(fair_df)
        + "\n\n## Scaling (synthetic households)\n\n" + md(scale_summary) + "\n"
    )
    figures(main_rows, fair_rows, scale_df, cap, day_for_plot=int(days[0]) if args.quick else 10)
    print((OUT / "tables.md").read_text())
    print(summary_json)


if __name__ == "__main__":
    main()
