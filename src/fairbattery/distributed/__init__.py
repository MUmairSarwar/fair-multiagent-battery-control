"""Distributed optimisation for the shared-transformer battery problem.

- `problem`: the one-day problem, split by household
- `household`: what each household solves locally
- `centralized`: one-solver reference solutions (efficient, proportional, max-min)
- `dual`: dual decomposition with a congestion price
- `admm`: ADMM for the sharing problem, with an optional fairness objective
- `hard_limit`: the extension with a hard transformer limit (separate subpackage)
"""

from .admm import sharing_admm
from .centralized import solve as solve_centralized
from .dual import dual_decomposition
from .problem import DayProblem, from_env, scaled_problem

__all__ = ["DayProblem", "from_env", "scaled_problem", "solve_centralized", "dual_decomposition", "sharing_admm"]
