"""Compare local log-savings ADMM to its central CVXPY reference."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from fairbattery.distributed.datasets import scenario
from fairbattery.distributed.fairness import distributed_proportional,solve_fair,metrics
from fairbattery import data
p=scenario(date='2025-05-14')
x,_=solve_fair(p,'proportional');_,efficient=solve_fair(p,'efficient')
reference=float(-np.log(-p.costs(x)/p.capacities).sum())
r=distributed_proportional(p,rho=.1,max_iter=2000,tol=1e-6)
row=dict(date='2025-05-14',rho=.1,max_iter=2000,tol=1e-6,
         iterations=r.iterations,runtime_seconds=r.seconds,converged=r.converged,
         log_utility_gap=r.history[-1]['objective']-reference,
         messages_per_agent=r.messages_per_agent,scalars_per_agent=r.scalars_per_agent,
         **metrics(p,r.x,efficient['total_saving_eur']))
out=data.REPO_ROOT/'results'/'distributed'/'complete';out.mkdir(exist_ok=True,parents=True)
pd.DataFrame(r.history).to_csv(out/'proportional_admm_history.csv',index=False)
(out/'proportional_admm_summary.json').write_text(json.dumps(row,indent=2))
print(json.dumps(row,indent=2))
