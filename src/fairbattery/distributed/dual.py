"""Dual decomposition with a congestion price.

Relax the transformer constraint with a multiplier lam[t] >= 0 per
quarter-hour. The Lagrangian then splits into one LP per household,

    min  (a + lam) . c_i + (b - lam) . d_i      (household i, on its own),

plus the coordinator's term for the overload variable o >= 0,
(penalty - lam) . o, which is bounded below only if lam <= penalty. So lam
lives in [0, penalty] (soft limit) or [0, inf) (hard limit).

lam is a *congestion price*: an extra EUR/kWh that each household pays for
importing in that quarter-hour. One iteration:

  1. the coordinator broadcasts lam (96 numbers),
  2. every household solves its own LP and sends back x_i = c_i - d_i,
  3. the coordinator moves lam along the constraint violation
         lam <- clip(lam + step_k * (sum_i x_i + base - limit - o), 0, lam_max)

For a linear program the household responses are piecewise constant in lam,
so the last iterate's schedule usually oscillates and can overload the
transformer. The standard fix, used here, is to report the running average of
the household schedules (ergodic primal recovery); it is feasible for every
household because their feasible sets are convex.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .household import LinearHousehold
from .problem import DayProblem


@dataclass
class DualResult:
    c: np.ndarray
    d: np.ndarray
    lam: np.ndarray
    history: dict = field(default_factory=dict)


def dual_decomposition(
    p: DayProblem,
    iters: int = 300,
    step0: float = 0.02,
    hard: bool = False,
    reference_cost: float | None = None,
) -> DualResult:
    n, T = p.n, p.T
    homes = [LinearHousehold(p, i) for i in range(n)]
    lam = np.zeros(T)
    lam_max = np.inf if hard else p.penalty
    c_sum, d_sum = np.zeros((n, T)), np.zeros((n, T))
    hist = {k: [] for k in ["dual_bound", "primal_cost_avg", "overload_last", "overload_avg", "gap_rel"]}
    best_dual = -np.inf
    for k in range(1, iters + 1):
        c, d = np.zeros((n, T)), np.zeros((n, T))
        dual_val = float(lam @ (p.base - p.limit))
        for i, h in enumerate(homes):
            c[i], d[i], f = h.solve(lam)
            dual_val += f
        best_dual = max(best_dual, dual_val)
        c_sum += c
        d_sum += d
        c_avg, d_avg = c_sum / k, d_sum / k

        total = (c - d).sum(axis=0) + p.base
        o = np.clip(total - p.limit, 0, None) if not hard else 0.0
        # o is only chosen > 0 once lam has reached the penalty
        o = np.where(lam >= lam_max - 1e-12, o, 0.0) if not hard else 0.0
        g = total - p.limit - o
        lam = np.clip(lam + step0 / np.sqrt(k) * g, 0, lam_max)

        primal = p.system_cost(c_avg, d_avg)
        hist["dual_bound"].append(best_dual)
        hist["primal_cost_avg"].append(primal)
        hist["overload_last"].append(float(p.overload(c - d).sum()))
        hist["overload_avg"].append(float(p.overload(c_avg - d_avg).sum()))
        ref = reference_cost if reference_cost is not None else best_dual
        hist["gap_rel"].append(abs(primal - ref) / max(abs(ref), 1e-9))
    return DualResult(c=c_avg, d=d_avg, lam=lam, history=hist)
