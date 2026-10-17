"""Turn results/runs/*.json into the tables and figures used in the README.

    python scripts/make_report.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"

LABEL = {"no_battery": "No battery", "rule": "Price rule", "lp": "LP optimum (perfect foresight)",
         "iql": "Independent Q-learning", "ippo": "Independent PPO", "mappo": "MAPPO"}
VARIANT_LABEL = {"main": "Main setting", "fairness": "+ fairness reward",
                 "no_incentive": "No overload price for agents", "communication": "+ 1-bit communication"}
COLOR = {"iql": "#1baf7a", "ippo": "#2a78d6", "mappo": "#eb6834", "rule": "#8a8986", "lp": "#0b0b0b",
         "no_battery": "#c3c2b7"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})


def load_runs():
    rows, logs, details = [], {}, {}
    for p in sorted((RES / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        m = r["metrics"]
        rows.append(m)
        logs[(m["variant"], m["algo"], m["seed"])] = [x["return"] for x in r["log"]]
        details[(m["variant"], m["algo"], m["seed"])] = r["detail"]
    return pd.DataFrame(rows), logs, details


def fmt(mean, std, nd=1):
    return f"{mean:.{nd}f} ± {std:.{nd}f}"


def main_table(df, base):
    lp_net = base["main"]["lp"]["metrics"]["net_benefit_eur"]
    cols = [("saving_eur", "Energy cost saved (EUR)", 2), ("overload_steps", "Overload steps", 0),
            ("overload_kwh", "Overload energy (kWh)", 1), ("net_benefit_eur", "Net benefit (EUR)", 2),
            ("share_of_lp", "Share of LP benefit (%)", 1), ("peak_kw", "Peak load (kW)", 2),
            ("mean_daily_peak_kw", "Mean daily peak (kW)", 2), ("jain", "Jain index", 3)]
    out = []
    for name in ["no_battery", "rule", "lp"]:
        m = dict(base["main"][name]["metrics"])
        m["share_of_lp"] = 100 * m["net_benefit_eur"] / lp_net
        row = {"Method": LABEL[name], "Seeds": "-"}
        for k, title, nd in cols:
            v = m[k]
            row[title] = "n/a" if (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"
        out.append(row)
    d = df[df.variant == "main"].copy()
    d["share_of_lp"] = 100 * d["net_benefit_eur"] / lp_net
    for algo in ["iql", "ippo", "mappo"]:
        g = d[d.algo == algo]
        row = {"Method": LABEL[algo], "Seeds": str(len(g))}
        for k, title, nd in cols:
            row[title] = fmt(g[k].mean(), g[k].std(ddof=1), nd)
        out.append(row)
    return pd.DataFrame(out)


def variant_table(df, base):
    lp_net = base["main"]["lp"]["metrics"]["net_benefit_eur"]
    d = df.copy()
    d["share_of_lp"] = 100 * d["net_benefit_eur"] / lp_net
    out = []
    for v in ["main", "fairness", "no_incentive", "communication"]:
        for algo in ["iql", "ippo", "mappo"]:
            g = d[(d.variant == v) & (d.algo == algo)]
            if g.empty:
                continue
            out.append({
                "Setting": VARIANT_LABEL[v], "Method": LABEL[algo],
                "Energy cost saved (EUR)": fmt(g.saving_eur.mean(), g.saving_eur.std(ddof=1), 2),
                "Overload steps": fmt(g.overload_steps.mean(), g.overload_steps.std(ddof=1), 0),
                "Net benefit (EUR)": fmt(g.net_benefit_eur.mean(), g.net_benefit_eur.std(ddof=1), 2),
                "Share of LP benefit (%)": fmt(g.share_of_lp.mean(), g.share_of_lp.std(ddof=1), 1),
                "Jain index": fmt(g.jain.mean(), g.jain.std(ddof=1), 3),
                "Lowest saving per kWh of battery (EUR)": fmt(g.min_saving_per_kwh_cap.mean(),
                                                               g.min_saving_per_kwh_cap.std(ddof=1), 2),
            })
    return pd.DataFrame(out)


def fig_net_benefit(df, base):
    d = df[df.variant == "main"]
    names = ["rule", "iql", "ippo", "mappo", "lp"]
    vals, errs, seeds = [], [], []
    for n in names:
        if n in ("rule", "lp"):
            vals.append(base["main"][n]["metrics"]["net_benefit_eur"])
            errs.append(0)
            seeds.append(None)
        else:
            g = d[d.algo == n].net_benefit_eur
            vals.append(g.mean())
            errs.append(g.std(ddof=1))
            seeds.append(g.to_numpy())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, height=0.55, color=[COLOR[n] for n in names], xerr=errs,
            error_kw={"ecolor": MUTED, "capsize": 3, "lw": 1})
    for yi, s in zip(y, seeds):
        if s is not None:
            ax.plot(s, [yi] * len(s), "o", ms=4, mfc="white", mec=INK, mew=0.8, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + (6 if v >= 0 else -6), yi, f"{v:.0f}", va="center",
                ha="left" if v >= 0 else "right", color=INK, fontsize=9)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_yticks(y, [LABEL[n] for n in names])
    ax.set_xlabel("Net benefit on 51 test days (EUR) = cost saved − 0.5 EUR × overload kWh")
    ax.set_title("Net benefit compared with no battery (bars: mean of 3 seeds, dots: single seeds)",
                 fontsize=10, loc="left", color=INK)
    lo = min(vals) - 60
    ax.set_xlim(lo, max(vals) + 40)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "net_benefit.png")
    plt.close(fig)


def fig_example_day(base, details):
    b = base["main"]
    price = np.array(b["price_eur_mwh"])
    spread = price.max(1) - price.min(1)
    day = int(np.argmax(spread))
    t = np.arange(96) / 4
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True,
                                 gridspec_kw={"height_ratios": [1, 2.2]})
    a1.step(t, price[day], where="post", color=MUTED, lw=1.5)
    a1.set_ylabel("Price (EUR/MWh)")
    a1.set_title(f"Test day {b['test_days'][day]}: day-ahead price (top) and total transformer load (bottom)",
                 fontsize=10, loc="left", color=INK)
    lim = 11.0
    a2.plot(t, np.array(b["base_load_kw"])[day], color=COLOR["no_battery"], lw=2, label="No battery")
    a2.plot(t, np.array(b["rule"]["detail"]["total_kw"])[day], color=COLOR["rule"], lw=2, label="Price rule")
    a2.plot(t, np.array(details[("main", "ippo", 0)]["total_kw"])[day], color=COLOR["ippo"], lw=2,
            label="Independent PPO (seed 0)")
    a2.plot(t, np.array(details[("main", "mappo", 0)]["total_kw"])[day], color=COLOR["mappo"], lw=2,
            label="MAPPO (seed 0)")
    a2.plot(t, np.array(b["lp"]["detail"]["total_kw"])[day], color=COLOR["lp"], lw=1.5, ls="--",
            label="LP optimum")
    a2.axhline(lim, color="#e34948", lw=1, ls=":")
    a2.text(0.2, lim + 0.4, "transformer limit 11 kW", color="#e34948", fontsize=9)
    a2.set_ylabel("Total load (kW)")
    a2.set_xlabel("Hour of day (local time)")
    a2.set_xticks(range(0, 25, 3))
    a2.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "example_day.png")
    plt.close(fig)


def fig_learning(logs):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for algo in ["iql", "ippo", "mappo"]:
        curves = [np.array(v) for (var, a, s), v in logs.items() if var == "main" and a == algo]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        c = np.stack([x[:n] for x in curves])
        w = max(1, n // 40)
        kern = np.ones(w) / w
        sm = np.stack([np.convolve(x, kern, mode="valid") for x in c])
        x = (np.arange(sm.shape[1]) + w) * 16
        ax.plot(x, sm.mean(0), color=COLOR[algo], lw=2, label=LABEL[algo])
        ax.fill_between(x, sm.min(0), sm.max(0), color=COLOR[algo], alpha=0.15, lw=0)
    ax.set_xlabel("Training days simulated")
    ax.set_ylabel("Training reward per day\n(sum over agents, scaled)")
    ax.set_title("Learning curves, main setting (line: mean of 3 seeds, band: min-max)",
                 fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "learning_curves.png")
    plt.close(fig)


def fig_variants(df):
    variants = ["main", "fairness", "no_incentive", "communication"]
    metrics = [("net_benefit_eur", "Net benefit (EUR)"), ("overload_steps", "Overload steps"),
               ("jain", "Jain index")]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    w = 0.25
    x = np.arange(len(variants))
    for ax, (k, title) in zip(axes, metrics):
        for j, algo in enumerate(["iql", "ippo", "mappo"]):
            means, stds = [], []
            for v in variants:
                g = df[(df.variant == v) & (df.algo == algo)][k]
                means.append(g.mean())
                stds.append(g.std(ddof=1))
            ax.bar(x + (j - 1) * w, means, w * 0.92, yerr=stds, color=COLOR[algo], label=LABEL[algo],
                   error_kw={"ecolor": MUTED, "capsize": 2, "lw": 0.8})
        ax.set_xticks(x, ["main", "+fairness", "no overload\nprice", "+1-bit\ncomm."], fontsize=8.5)
        ax.set_title(title, fontsize=10, loc="left", color=INK)
        ax.grid(axis="x", visible=False)
        if k == "jain":
            lo = np.nanmin(df[k]) - 0.02
            ax.set_ylim(max(0, lo), 1.0)
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Experiments (mean ± std over 3 seeds)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "variants.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    base = json.loads((RES / "baselines.json").read_text())
    df, logs, details = load_runs()
    if df.empty:
        sys.exit("no runs found in results/runs")
    df.drop(columns=[], errors="ignore").to_csv(RES / "all_runs.csv", index=False)
    t1 = main_table(df, base)
    t2 = variant_table(df, base)
    t1.to_csv(RES / "table_main.csv", index=False)
    t2.to_csv(RES / "table_experiments.csv", index=False)
    (RES / "table_main.md").write_text(t1.to_markdown(index=False))
    (RES / "table_experiments.md").write_text(t2.to_markdown(index=False))
    lp = base["main"]["lp"]["metrics"]
    extra = {
        "baseline_energy_cost_eur": round(base["main"]["rule"]["metrics"]["saving_eur"] / (base["main"]["rule"]["metrics"]["saving_pct"] / 100), 2),
        "lp_simultaneous_charge_discharge_steps": lp.get("simultaneous_charge_discharge_steps"),
        "n_test_days": lp["n_days"], "n_test_steps": lp["n_steps"],
    }
    (RES / "notes.json").write_text(json.dumps(extra, indent=2))
    fig_net_benefit(df, base)
    fig_example_day(base, details)
    fig_learning(logs)
    fig_variants(df)
    print(t1.to_markdown(index=False))
    print()
    print(t2.to_markdown(index=False))


if __name__ == "__main__":
    main()
