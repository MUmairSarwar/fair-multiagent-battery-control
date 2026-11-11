"""Convex hard-capacity battery model, using kWh per interval throughout."""
from dataclasses import dataclass
import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
from ...lp_variants import sufficient_throughput_cost


@dataclass
class DayProblem:
    prices: np.ndarray
    loads: np.ndarray  # N x T, kWh
    capacities: np.ndarray
    power: np.ndarray  # kW
    limit: np.ndarray  # T, kWh
    eta: float = .95
    soc_init: float = .5
    dt: float = .25
    fee: float | None = None

    def __post_init__(self):
        for k in ('prices','loads','capacities','power','limit'):
            setattr(self,k,np.asarray(getattr(self,k),dtype=float).copy())
        self.n,self.T=self.loads.shape
        if self.prices.shape!=(self.T,) or self.limit.shape!=(self.T,):
            raise ValueError('prices and limit must have one entry per interval')
        if self.capacities.shape!=(self.n,) or self.power.shape!=(self.n,):
            raise ValueError('one capacity and power bound per household required')
        if not all(np.isfinite(getattr(self,k)).all() for k in ('prices','loads','capacities','power','limit')):
            raise ValueError('nonfinite data')
        if np.any(self.loads<0) or np.any(self.capacities<=0) or np.any(self.power<=0) or np.any(self.limit<=0):
            raise ValueError('loads nonnegative; capacities, powers and limits positive')
        if not 0<self.eta<=1 or not 0<=self.soc_init<=1 or self.dt<=0:
            raise ValueError('invalid physical parameters')
        if self.fee is None:
            self.fee=sufficient_throughput_cost(self.prices,self.eta)
        if not np.isfinite(self.fee) or self.fee<0:
            raise ValueError('fee must be finite and nonnegative')

    @classmethod
    def from_env(cls,env,b=0):
        return cls(env.price[b],env.loads[b],env.cap,env.pmax,
                   np.full(env.price.shape[1],env.cfg.transformer_kw*.25),
                   eta=env.eta,soc_init=env.cfg.soc_init)

    @property
    def cost(self):
        p=self.prices; ref=p.mean()
        return np.r_[p-ref*self.eta+self.fee,-p+ref/self.eta+self.fee,np.zeros(self.T)]

    @property
    def G(self):
        I=sp.eye(self.T,format='csc')
        return sp.hstack([I,-I,sp.csc_matrix((self.T,self.T))],format='csc')

    def local(self,i):
        T=self.T; I=sp.eye(T,format='csc')
        prev=sp.diags(np.ones(T-1),-1,shape=(T,T),format='csc')
        A=sp.hstack([-self.eta*I,I/self.eta,I-prev],format='csc')
        b=np.zeros(T); b[0]=self.soc_init*self.capacities[i]
        ub=np.r_[np.full(T,self.power[i]*self.dt),np.minimum(self.power[i]*self.dt,self.loads[i]),np.full(T,self.capacities[i])]
        return A,b,ub

    def idle(self):
        x=np.zeros((self.n,3*self.T));x[:,2*self.T:]=self.soc_init*self.capacities[:,None]
        return x

    def grid(self,x):
        return self.loads+x[:,:self.T]-x[:,self.T:2*self.T]

    def costs(self,x):
        return x@self.cost

    def clean(self,x):
        """Remove simultaneous charge and discharge without changing stored energy.

        Only applied when the fee is sufficient, because only then is the
        cost guaranteed to fall (theory, Section 9). Grid import falls too.
        """
        x=x.copy();T=self.T
        if self.fee+1e-12 < sufficient_throughput_cost(self.prices,self.eta,margin=1e-12):
            return x
        c=np.maximum(x[:,:T],0);d=np.maximum(x[:,T:2*T],0)
        loop=np.minimum(c,d/self.eta**2)
        x[:,:T]=c-loop;x[:,T:2*T]=d-self.eta**2*loop
        x[:,2*T:]=self.soc_init*self.capacities[:,None]+np.cumsum(self.eta*x[:,:T]-x[:,T:2*T]/self.eta,axis=1)
        return x

    def recover(self,x):
        """Clean ``x`` and blend it with the idle schedule until the hard limit holds.

        Returns the schedule and the blend factor, the only number the
        coordinator has to broadcast. Requires the idle schedule to be feasible.
        """
        x=self.clean(x)
        baseline=self.loads.sum(axis=0)
        if np.any(baseline>self.limit+1e-10):
            raise ValueError('Idle-feasible recovery required; supply a different feasible reference for this dataset')
        delta=self.grid(x).sum(axis=0)-baseline
        positive=delta>1e-12
        ratio=np.ones(self.T)
        ratio[positive]=np.minimum(1,(self.limit[positive]-baseline[positive])/delta[positive])
        retain=float(np.clip(ratio.min(),0,1))
        return self.idle()+retain*(x-self.idle()),retain

    def diagnostics(self,x):
        T=self.T;c=x[:,:T];d=x[:,T:2*T];e=x[:,2*T:]
        expected=self.soc_init*self.capacities[:,None]+np.cumsum(self.eta*c-d/self.eta,axis=1)
        violation=max(0,float(-x.min()),float((c-self.power[:,None]*self.dt).max()),
                      float((d-np.minimum(self.loads,self.power[:,None]*self.dt)).max()),
                      float((e-self.capacities[:,None]).max()),float(np.abs(e-expected).max()))
        return dict(objective=float(self.costs(x).sum()),
                    max_capacity_violation_kw=float(np.maximum(self.grid(x).sum(axis=0)-self.limit,0).max()/self.dt),
                    local_violation_kwh=violation,
                    overlap_steps=int(((c>1e-6)&(d>1e-6)).sum()))


def central_lp(problem: DayProblem):
    """Central reference: sparse HiGHS LP with the throughput fee and the hard limit."""
    parts=[problem.local(i) for i in range(problem.n)]
    Aeq=sp.block_diag([p[0] for p in parts],format='csc')
    beq=np.concatenate([p[1] for p in parts]);ub=np.concatenate([p[2] for p in parts])
    G=sp.hstack([problem.G]*problem.n,format='csc')
    result=linprog(np.tile(problem.cost,problem.n),A_ub=G,
                   b_ub=problem.limit-problem.loads.sum(axis=0),A_eq=Aeq,b_eq=beq,
                   bounds=np.c_[np.zeros(len(ub)),ub],method='highs')
    if not result.success: raise RuntimeError(result.message)
    return result.x.reshape(problem.n,-1),float(result.fun)


def cvxpy_model(problem):
    """Separate CVXPY formulation, used to check the LP and for the fairness models."""
    import cvxpy as cp
    n,T=problem.n,problem.T
    c=cp.Variable((n,T),nonneg=True);d=cp.Variable((n,T),nonneg=True)
    e=cp.Variable((n,T))
    previous=cp.hstack([problem.soc_init*problem.capacities[:,None],e[:,:-1]]) if T>1 else problem.soc_init*problem.capacities[:,None]
    constraints=[e==previous+problem.eta*c-d/problem.eta,e>=0,e<=problem.capacities[:,None],
                 c<=problem.power[:,None]*problem.dt,d<=np.minimum(problem.loads,problem.power[:,None]*problem.dt),
                 cp.sum(problem.loads+c-d,axis=0)<=problem.limit]
    q=problem.cost
    costs=cp.sum(cp.multiply(q[:T],c)+cp.multiply(q[T:2*T],d),axis=1)
    return c,d,e,costs,constraints


def central_cvxpy(problem):
    import cvxpy as cp
    c,d,e,costs,constraints=cvxpy_model(problem)
    task=cp.Problem(cp.Minimize(cp.sum(costs)),constraints)
    task.solve(solver='CLARABEL',canon_backend='SCIPY',tol_gap_abs=1e-8,tol_feas=1e-8,tol_gap_rel=1e-8)
    if task.status!='optimal': raise RuntimeError(task.status)
    return np.hstack([c.value,d.value,e.value]),float(task.value)
