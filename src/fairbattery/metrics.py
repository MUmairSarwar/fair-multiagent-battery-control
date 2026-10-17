"""Rollouts and the metrics used in the results table."""

from __future__ import annotations

import numpy as np

from .data import STEPS_PER_DAY
from .env import BatteryCommunity


def jain_index(x: np.ndarray) -> float:
    """Jain's fairness index (sum x)^2 / (n * sum x^2).

    It is 1 when everyone gets the same and 1/n when one person gets
    everything. It is only meaningful for non-negative values, so negative
    savings are set to zero first (the table also reports the smallest saving).
    """
    x = np.clip(np.asarray(x, dtype=float), 0, None)
    if np.all(x == 0):
        return float("nan")
    return float(x.sum() ** 2 / (len(x) * (x**2).sum()))


def rollout(env: BatteryCommunity, policy, day_idx, rng, loads=None) -> dict:
    """Run a policy over a batch of days. Returns arrays with a step axis."""
    obs = env.reset(day_idx, rng, loads=loads)
    keys = ["saving", "total_kw", "overload_kwh", "grid_kwh"]
    out = {k: [] for k in keys}
    done = False
    while not done:
        a = policy.act(env, obs)
        obs, _, done, info = env.step(a)
        for k in keys:
            out[k].append(info[k])
    res = {k: np.stack(v, axis=-1) for k, v in out.items()}  # step axis last
    return res


def summarise(res: dict, env: BatteryCommunity, base_cost: float) -> dict:
    """Metrics over a set of evaluation days.

    res["saving"]: [B, n, T]; res["total_kw"], res["overload_kwh"]: [B, T].
    """
    cfg = env.cfg
    saving_house = res["saving"].sum(axis=(0, 2))  # EUR per household
    total_saving = float(saving_house.sum())
    overload_kwh = float(res["overload_kwh"].sum())
    benefit = total_saving - cfg.overload_penalty * overload_kwh
    per_cap = saving_house / env.cap
    return {
        "saving_eur": total_saving,
        "saving_pct": 100 * total_saving / base_cost,
        "net_benefit_eur": benefit,
        "peak_kw": float(res["total_kw"].max()),
        "mean_daily_peak_kw": float(res["total_kw"].max(axis=1).mean()),
        "overload_steps": int((res["overload_kwh"] > 1e-9).sum()),
        "overload_kwh": overload_kwh,
        "jain": jain_index(per_cap),
        "min_saving_per_kwh_cap": float(per_cap.min()),
        "max_saving_per_kwh_cap": float(per_cap.max()),
        "n_days": int(res["total_kw"].shape[0]),
        "n_steps": int(res["total_kw"].shape[0] * STEPS_PER_DAY),
    }


def baseline_cost(env: BatteryCommunity) -> float:
    """Energy cost of all households with no battery for the current batch."""
    return float((env.price[:, None, :] * env.loads).sum())
