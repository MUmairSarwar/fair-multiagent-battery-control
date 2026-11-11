"""Shared-transformer problem with a hard limit (docs/theory.md, Sections 12-16).

- `model`: the daily convex model, central reference LP and CVXPY formulation
- `local`: the subproblems each household solves
- `algorithms`: dual decomposition with primal recovery, and sharing ADMM
- `fairness`: efficient, proportional and lexicographic max-min references,
  and proportional fairness inside ADMM
- `datasets`: fixed-seed scenarios for 6 to 1,000 households
"""
