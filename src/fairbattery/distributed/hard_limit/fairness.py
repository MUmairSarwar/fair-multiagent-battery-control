"""Efficient, proportional and max-min fair schedules under the hard limit.

Fairness is measured on each household's daily saving per kWh of battery,
relative to an idle battery, including the throughput fee and the end-of-day
valuation. Negative savings are rejected, not clipped.
"""
import numpy as np
import cvxpy as cp
from .model import cvxpy_model


def solve_fair(problem,kind='proportional',lex_tol=2e-7):
    c,d,e,costs,cons=cvxpy_model(problem)
    normalized=cp.multiply(1/problem.capacities,-costs)
    # No household may end up worse off than with an idle battery.
    cons=list(cons)+[normalized>=0]
    solve_options=dict(solver='CLARABEL',tol_gap_abs=1e-7,tol_feas=1e-8,tol_gap_rel=1e-7,max_iter=300,max_step_fraction=.9)
    def solve(objective,extra=()):
        task=cp.Problem(objective,cons+list(extra));task.solve(**solve_options,canon_backend='SCIPY')
        if task.status!='optimal':raise RuntimeError('Fairness solve: '+task.status)
        return float(task.value)
    if kind=='efficient':
        solve(cp.Minimize(cp.sum(costs)))
    elif kind=='proportional':
        # The log needs every saving > 0: check that this is achievable instead of adding an offset.
        floor=cp.Variable()
        achievable=solve(cp.Maximize(floor),[normalized>=floor])
        if achievable<=1e-8:
            raise ValueError('Strictly positive savings for every household are infeasible; log fairness is undefined')
        solve(cp.Maximize(cp.sum(cp.log(normalized / achievable))))
    elif kind=='max_min':
        active=list(range(problem.n));frozen=[]
        while active:
            floor=cp.Variable()
            level=solve(cp.Maximize(floor),frozen+[normalized[active]>=floor])
            face=frozen+[normalized[active]>=level-lex_tol]
            # Freeze the households that cannot rise above the current floor.
            blocked=[]
            for i in active:
                attainable=solve(cp.Maximize(normalized[i]),face)
                if attainable<=level+10*lex_tol:blocked.append(i)
            if not blocked:
                raise RuntimeError('No max-min bottleneck identified within numerical tolerance')
            frozen += [normalized[blocked]>=level-lex_tol,
                       normalized[blocked]<=level+lex_tol]
            active=[i for i in active if i not in blocked]
        # Among the lexicographically fair schedules, take the most efficient one.
        solve(cp.Minimize(cp.sum(costs)),frozen)
    else:raise ValueError(kind)
    x=np.hstack([c.value,d.value,e.value])
    # No cleaning step here: it could move the frozen max-min levels.
    return x,metrics(problem,x)


def metrics(problem,x,efficient_benefit=None):
    savings=-problem.costs(x);q=savings/problem.capacities
    if q.min() < -1e-6:raise ValueError('Negative savings: Jain index not reported by clipping')
    # Allow solver round-off; Jain's index is undefined when every saving is zero.
    jain=float(q.sum()**2/(len(q)*(q@q))) if q@q>1e-20 else None
    total=float(savings.sum())
    return dict(total_saving_eur=total,saving_per_kwh=q.tolist(),jain=jain,
                min_saving_per_kwh=float(q.min()),
                price_of_fairness=(efficient_benefit-total)/efficient_benefit
                if efficient_benefit is not None and efficient_benefit>1e-12 else None,
                **problem.diagnostics(x))


def distributed_proportional(p,rho=10.,max_iter=1000,tol=1e-5):
    """Proportional fairness with sharing ADMM.

    Each household maximises the log of its own saving plus the ADMM penalty;
    the coordinator step is the same projection as in ``sharing_admm``.
    Max-min fairness has no distributed version; see ``solve_fair``.
    """
    from time import perf_counter
    from .local import HouseholdLogQP
    from .algorithms import DistributedResult
    start=perf_counter();agents=[HouseholdLogQP(p,i,rho) for i in range(p.n)]
    z=p.loads.copy();u=np.zeros(p.T);best=None;value=np.inf;history=[];converged=False
    p.recover(p.idle())
    for k in range(1,max_iter+1):
        answers=[a.solve(z[i],u) for i,a in enumerate(agents)]
        x=np.stack([a[0] for a in answers]);g=np.stack([a[1] for a in answers])
        v=g+u;delta=np.maximum((v.sum(axis=0)-p.limit)/p.n,0);zn=v-delta
        r=g-zn;dz=zn-z;s=rho*np.hstack([dz,-dz,np.zeros_like(dz)])
        rn=float(np.linalg.norm(r));sn=float(np.linalg.norm(s));z=zn;u=delta
        feasible,_=p.recover(x);q=-p.costs(feasible)/p.capacities
        if np.all(q>0):
            phi=float(-np.log(q).sum())
            if phi<value:value=phi;best=feasible
        if k==1 or k%10==0 or k==max_iter:
            rr=rn/np.sqrt(p.n*p.T);ss=sn/np.sqrt(3*p.n*p.T)
            history.append(dict(iteration=k,objective=value,primal_residual=rn,dual_residual=sn,
                                primal_rms=rr,dual_rms=ss,raw_violation_kw=p.diagnostics(x)['max_capacity_violation_kw']))
            if rr<=tol and ss<=tol and best is not None:
                converged=True;break
    if best is None:raise RuntimeError('No positive feasible saving schedule recovered')
    return DistributedResult(best,history,k,perf_counter()-start,converged,2*k,k*(2*p.T+3))
