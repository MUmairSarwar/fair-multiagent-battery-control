"""Run the hard-limit experiments (dual, ADMM, sensitivity, fairness, scaling) with fixed seeds."""
import argparse,json,sys,platform,hashlib
from pathlib import Path
from time import perf_counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import scipy,cvxpy,highspy,osqp
from fairbattery.distributed.datasets import scenario
from fairbattery.distributed.model import central_lp,central_cvxpy
from fairbattery.distributed.algorithms import dual_decomposition,sharing_admm
from fairbattery.distributed.fairness import solve_fair,metrics
from fairbattery import data


def record_run(p,method,reference,tag,out,**kwargs):
    print('START',tag,flush=True)
    fn=dual_decomposition if method=='dual' else sharing_admm
    result=fn(p,reference=reference,**kwargs)
    diag=p.diagnostics(result.x)
    row=dict(tag=tag,algorithm=method,n=p.n,T=p.T,iterations=result.iterations,
             runtime_seconds=result.seconds,converged=result.converged,
             objective_gap=diag['objective']-reference,
             objective_gap_per_agent=(diag['objective']-reference)/p.n,
             relative_objective_gap=(diag['objective']-reference)/max(abs(reference),1e-12),
             messages_per_agent=result.messages_per_agent,scalars_per_agent=result.scalars_per_agent,
             **diag,**kwargs)
    pd.DataFrame(result.history).to_csv(out/f'{tag}_history.csv',index=False)
    (out/f'{tag}_summary.json').write_text(json.dumps(row,indent=2))
    print('DONE',json.dumps(row),flush=True)
    return row


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=data.REPO_ROOT/'results'/'distributed'/'complete')
    ap.add_argument('--sizes',type=int,nargs='+',default=[6,100,1000])
    ap.add_argument('--skip-scaling',action='store_true')
    ap.add_argument('--only-scaling',action='store_true')
    args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    dates=['2025-01-07','2025-05-14','2025-08-06'];comparisons=[];fair=[]
    if not args.only_scaling:
        for date in dates:
            p=scenario(date=date);t=perf_counter();_,reference=central_lp(p);secs=perf_counter()-t
            _,cv=central_cvxpy(p)
            if abs(cv-reference)>1e-6:raise AssertionError('Central solvers disagree')
            comparisons.append(dict(tag=date+'_central',algorithm='central_lp',n=6,T=96,
                                    objective=reference,runtime_seconds=secs,cvxpy_objective=cv))
            for method,params in [('dual',dict(step=.01,max_iter=3000,tol=5e-4)),
                                  ('admm',dict(rho=.1,max_iter=2000,tol=1e-5))]:
                row=record_run(p,method,reference,date+'_'+method,out,**params)
                comparisons.append(row)
            xe,me=solve_fair(p,'efficient');benefit=me['total_saving_eur']
            for kind in ['efficient','proportional','max_min']:
                if kind=='efficient':x=xe
                else:x,_=solve_fair(p,kind)
                m=metrics(p,x,benefit);q=m.pop('saving_per_kwh')
                m.update({f'household_{i+1}_saving_per_kwh':v for i,v in enumerate(q)})
                fair.append(dict(date=date,fairness=kind,**m))
            pd.DataFrame(comparisons).to_csv(out/'comparisons.csv',index=False)
            pd.DataFrame(fair).to_csv(out/'fairness.csv',index=False)
        p=scenario();_,reference=central_lp(p);sweep=[]
        for step in [.01,.03,.1]:
            sweep.append(record_run(p,'dual',reference,f'dual_step_{step}',out,step=step,max_iter=2000,tol=5e-4))
        # Constant step for comparison only; it is not expected to converge exactly.
        sweep.append(record_run(p,'dual',reference,'dual_constant_0.03',out,step=.03,rule='constant',max_iter=2000,tol=5e-4))
        for rho in [.01,.1,1.]:
            sweep.append(record_run(p,'admm',reference,f'admm_rho_{rho}',out,rho=rho,max_iter=2000,tol=1e-5))
        pd.DataFrame(sweep).to_csv(out/'sensitivity.csv',index=False)
    if not args.skip_scaling:
        scaling=[]
        for n in args.sizes:
            p=scenario(n=n);t=perf_counter();_,reference=central_lp(p);secs=perf_counter()-t
            scaling.append(dict(algorithm='central_lp',n=n,T=96,objective=reference,runtime_seconds=secs))
            for method,params in [('dual',dict(step=.01,max_iter=1200,tol=1e-3)),
                                  ('admm',dict(rho=.1,max_iter=1000,tol=1e-5))]:
                row=record_run(p,method,reference,f'scale_{n}_{method}',out,**params)
                scaling.append(row)
                pd.DataFrame(scaling).to_csv(out/'scaling.csv',index=False)
    metadata=dict(seed=2025,dates=dates,model='hard capacity; sufficient throughput fee; terminal mean-price valuation',
                  capacity_scaling='11*N/6 kW',execution='serial simulation of separate household solvers',
                  versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                                cvxpy=cvxpy.__version__,highspy=highspy.Highs().version(),osqp=osqp.__version__),
                  input_sha256=hashlib.sha256(data.PROCESSED_FILE.read_bytes()).hexdigest(),
                  dual_stopping='feasible primal minus best dual <= tol*N EUR',
                  admm_stopping='primal RMS and dual RMS <= tol; objective gap independently reported',
                  communication='two logical messages per agent per iteration, 2*T+3 scalars; diagnostics included; no network timing')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
