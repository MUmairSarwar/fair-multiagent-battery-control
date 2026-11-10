"""Centralised reference solutions (one solver sees everything).

These are the ground truth the distributed algorithms are compared with.
`efficient_soft` is the same problem as the LP in `baselines.py`.
The fairness problems use a hard transformer limit, so that every allocation
compared is overload-free and the comparison is only about who gets what.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from ..data import DT_HOURS
from .problem import DayProblem


def _variables(p: DayProblem):
    n, T = p.n, p.T
    C, D = cp.Variable((n, T)), cp.Variable((n, T))
    cmax = (p.pmax * DT_HOURS)[:, None]
    E = p.soc_init * p.cap[:, None] + cp.cumsum(p.eta * C - D / p.eta, axis=1)
    cons = [C >= 0, D >= 0, C <= cmax, D <= np.minimum(cmax, p.loads), E >= 0, E <= p.cap[:, None]]
    saving = -(C @ p.a + D @ p.b)  # [n]
    return C, D, cons, saving


def solve(p: DayProblem, objective: str = "efficient_soft", eps: float = 1e-3):
    """objective: 'efficient_soft' | 'efficient' | 'proportional' | 'maxmin'."""
    C, D, cons, s = _variables(p)
    total = cp.sum(C - D, axis=0) + p.base
    u = cp.multiply(s, 1 / p.cap)  # saving per kWh of battery
    if objective == "efficient_soft":
        obj = cp.Minimize(-cp.sum(s) + p.penalty * cp.sum(cp.pos(total - p.limit)))
    else:
        cons = cons + [total <= p.limit]
        if objective == "efficient":
            obj = cp.Maximize(cp.sum(s))
        elif objective == "proportional":
            obj = cp.Maximize(p.cap @ cp.log(u + eps))
        elif objective == "maxmin":
            obj = cp.Maximize(cp.min(u))
        else:
            raise ValueError(objective)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.CLARABEL)
    if C.value is None:
        raise RuntimeError(f"{objective}: {prob.status}")
    c = np.clip(C.value, 0, None)
    d = np.clip(D.value, 0, None)
    return c, d
