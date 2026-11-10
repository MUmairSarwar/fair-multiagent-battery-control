"""Baselines: no battery, a simple price rule, and the perfect-foresight LP."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .data import DT_HOURS, STEPS_PER_DAY
from .env import IDLE_ACTION, BatteryCommunity


class NoBatteryPolicy:
    name = "no_battery"

    def act(self, env: BatteryCommunity, obs: np.ndarray) -> np.ndarray:
        return np.full((env.B, env.n), IDLE_ACTION)


class PriceRulePolicy:
    """Charge at full power in the cheapest quarter of the day, discharge in the
    most expensive quarter, otherwise do nothing. Every household follows the
    same rule and nobody looks at the transformer."""

    name = "rule"

    def __init__(self, low: float = 0.25, high: float = 0.75):
        self.low, self.high = low, high

    def act(self, env: BatteryCommunity, obs: np.ndarray) -> np.ndarray:
        rank = env.price_rank[:, env.t]  # [B]
        a = np.full(env.B, IDLE_ACTION)
        a[rank <= self.low] = 4  # charge at full power
        a[rank >= self.high] = 0  # discharge at full power
        return np.repeat(a[:, None], env.n, axis=1)


def _build_day_lp(env: BatteryCommunity, b: int):
    """Build the original LP coefficients for day `b` of the current batch.

    Objective: negative total saving plus overload penalty, with perfect
    knowledge of the day's prices and loads.

    Variables per household and step: c (kWh drawn from the grid to charge)
    and d (kWh delivered from the battery to the house). Plus one overload
    variable o per step.
    """
    n, T = env.n, STEPS_PER_DAY
    eta, cfg = env.eta, env.cfg
    price = env.price[b]
    p_ref = env.p_ref[b]
    load = env.loads[b]  # [n, T]
    nv = 2 * n * T + T

    def ci(i, t):
        return i * T + t

    def di(i, t):
        return n * T + i * T + t

    def oi(t):
        return 2 * n * T + t

    # objective: minimise sum price*(c-d) - p_ref*(eta*c - d/eta) + penalty*o
    cost = np.zeros(nv)
    for i in range(n):
        cost[i * T : (i + 1) * T] = price - p_ref * eta
        cost[n * T + i * T : n * T + (i + 1) * T] = -price + p_ref / eta
    cost[2 * n * T :] = cfg.overload_penalty

    bounds = []
    for i in range(n):
        bounds += [(0, env.pmax[i] * DT_HOURS)] * T
    for i in range(n):
        for t in range(T):
            bounds.append((0, min(env.pmax[i] * DT_HOURS, load[i, t])))
    bounds += [(0, None)] * T

    # state of charge after step t: e0 + sum_{k<=t} (eta*c_k - d_k/eta) in [0, cap]
    rows, cols, vals = [], [], []
    b_ub = []
    r = 0
    for i in range(n):
        e0 = cfg.soc_init * env.cap[i]
        for t in range(T):
            ks = np.arange(t + 1)
            # upper: e0 + sum <= cap
            rows += [r] * (2 * len(ks))
            cols += [ci(i, k) for k in ks] + [di(i, k) for k in ks]
            vals += [eta] * len(ks) + [-1 / eta] * len(ks)
            b_ub.append(env.cap[i] - e0)
            r += 1
            # lower: -(e0 + sum) <= 0
            rows += [r] * (2 * len(ks))
            cols += [ci(i, k) for k in ks] + [di(i, k) for k in ks]
            vals += [-eta] * len(ks) + [1 / eta] * len(ks)
            b_ub.append(e0)
            r += 1
    # transformer: sum_i (load + c - d) - o <= limit * dt
    for t in range(T):
        rows += [r] * (2 * n + 1)
        cols += [ci(i, t) for i in range(n)] + [di(i, t) for i in range(n)] + [oi(t)]
        vals += [1.0] * n + [-1.0] * n + [-1.0]
        b_ub.append(cfg.transformer_kw * DT_HOURS - load[:, t].sum())
        r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    return cost, A, np.asarray(b_ub), bounds


def solve_day_lp(env: BatteryCommunity, b: int, *, throughput_cost: float = 0.0):
    """Continuous relaxation; a fee discourages but need not prevent cycling.

    throughput_cost is EUR per AC-side kWh charged plus discharged.
    Use solve_day_milp when mutually exclusive modes must be guaranteed.
    """
    from .distributed.centralised import solve_day

    result = solve_day(env, b, throughput_cost=throughput_cost)
    return result.charge, result.discharge


def solve_day_milp(env: BatteryCommunity, b: int):
    """Physical continuous-power baseline with binary charge/discharge modes."""
    from .distributed.centralised import solve_day

    result = solve_day(env, b, exclusive=True)
    return result.charge, result.discharge


def run_lp(env: BatteryCommunity, *, throughput_cost: float = 0.0, exclusive: bool = False):
    """Solve each day and replay the original environment accounting.

    Default retains the legacy relaxed LP for reproducibility. exclusive=True
    selects the physically exclusive MILP. Savings do NOT subtract a modelling
    throughput fee; that fee and the optimized objective are returned separately.
    """
    from .distributed.centralised import solve_day
    B, n, T = env.B, env.n, STEPS_PER_DAY
    charge = np.zeros((B, n, T))
    discharge = np.zeros((B, n, T))
    solutions = []
    for b in range(B):
        result = solve_day(env, b, throughput_cost=throughput_cost, exclusive=exclusive)
        charge[b], discharge[b] = result.charge, result.discharge
        solutions.append(result)
    eta = env.eta
    e0 = env.cfg.soc_init * env.cap
    flow = charge * eta - discharge / eta
    energy = e0[None, :, None] + np.cumsum(flow, axis=2)
    e_prev = np.concatenate([np.repeat(e0[None, :, None], B, 0), energy[:, :, :-1]], axis=2)
    grid = env.loads + charge - discharge
    total_kw = grid.sum(axis=1) / DT_HOURS
    overload = np.clip(total_kw - env.cfg.transformer_kw, 0, None) * DT_HOURS
    saving = env.price[:, None, :] * (env.loads - grid) + env.p_ref[:, None, None] * (energy - e_prev)
    both = ((charge > 1e-6) & (discharge > 1e-6)).sum()
    return {
        "saving": saving,
        "charge_kwh": charge,
        "discharge_kwh": discharge,
        "energy_kwh": energy,
        "objective_eur": np.array([r.objective for r in solutions]),
        "throughput_fee_eur": np.array([r.throughput_fee for r in solutions]),
        "solver_seconds": np.array([r.solve_seconds for r in solutions]),
        "mip_gaps": np.array([r.mip_gap for r in solutions]),
        "total_kw": total_kw,
        "overload_kwh": overload,
        "grid_kwh": grid,
        "simultaneous_charge_discharge_steps": int(both),
    }


def run_milp(env: BatteryCommunity):
    """Evaluate the exclusive-mode baseline using the unchanged economic score."""
    return run_lp(env, exclusive=True)


def run_corrected_lp(env: BatteryCommunity):
    """Cycle-free penalized LP for this model; original scores remain separate.

    Uses one sufficient EUR/kWh fee over the entire current batch. This is a
    changed objective, not the unpenalized exclusive MILP optimum.
    """
    from .distributed.centralised import sufficient_throughput_cost

    fee = sufficient_throughput_cost(env.price, env.eta)
    result = run_lp(env, throughput_cost=fee)
    if result['simultaneous_charge_discharge_steps']:
        raise RuntimeError('Unexpected cycling in sufficient-fee LP')
    result['throughput_cost_eur_per_kwh'] = fee
    return result
