## Main comparison

| Method                                        |   Net benefit (EUR) |   Share of LP (%) |   Overload (kWh) | Iterations (median)   |
|:----------------------------------------------|--------------------:|------------------:|-----------------:|:----------------------|
| Centralised LP (published bound)              |              132.06 |            100    |            0     | -                     |
| Exact MILP (no simultaneous charge/discharge) |              132.05 |            100    |            0     | -                     |
| Dual decomposition (300 iterations, averaged) |              131.28 |             99.42 |            0.718 | 300                   |
| ADMM (rho = 0.5)                              |              132.05 |             99.99 |            0.01  | 177                   |

## Fairness (hard transformer limit)

| Allocation                                |   Total saving (EUR) |   Price of fairness (%) |   Jain index |   Lowest saving per kWh of battery (EUR) |   Highest saving per kWh of battery (EUR) |
|:------------------------------------------|---------------------:|------------------------:|-------------:|-----------------------------------------:|------------------------------------------:|
| Efficient (max total saving)              |               132.06 |                    0    |       0.9741 |                                    2.269 |                                     3.808 |
| Proportional fairness                     |               132.03 |                    0.02 |       0.9939 |                                    2.594 |                                     3.348 |
| Max-min fairness                          |               118.2  |                   10.49 |       1      |                                    2.624 |                                     2.631 |
| Proportional fairness, distributed (ADMM) |               132.03 |                    0.02 |       0.9938 |                                    2.592 |                                     3.352 |

## Scaling (synthetic households)

|   households |   admm_iters_to_1pct |   admm_final_gap_pct_max|   dual_iters_to_1pct |   dual_final_gap_pct_max|   admm_ms_per_household_iter |
|-------------:|---------------------:|---------------------:|---------------------:|---------------------:|-----------------------------:|
|            6 |                 60.5 |                0.007 |                  158 |                1.305 |                        3.812 |
|           24 |                 60   |                0.007 |                  nan |                7.689 |                        4.277 |
|           96 |                 60.5 |                0.006 |                  nan |               11.197 |                        4.081 |
