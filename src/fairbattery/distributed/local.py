"""Private household oracles. No oracle receives another household's data."""
import numpy as np
import scipy.sparse as sp
import highspy
import osqp


class HouseholdLP:
    def __init__(self,problem,i):
        self.T=problem.T;self.cost=problem.cost;self.load=problem.loads[i].copy()
        A,b,upper=problem.local(i);A=A.tocsc();nv=3*self.T
        lp=highspy.HighsLp();lp.num_col_=nv;lp.num_row_=self.T
        lp.col_cost_=self.cost;lp.col_lower_=np.zeros(nv);lp.col_upper_=upper
        lp.row_lower_=b;lp.row_upper_=b
        lp.a_matrix_.format_=highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_=A.indptr;lp.a_matrix_.index_=A.indices;lp.a_matrix_.value_=A.data
        self.h=highspy.Highs();self.h.setOptionValue('output_flag',False)
        self.h.setOptionValue('threads',1);self.h.setOptionValue('random_seed',0)
        self.h.setOptionValue('solver','simplex');self.h.passModel(lp)
        self.idx=np.arange(nv,dtype=np.int32)

    def solve(self,price):
        q=self.cost+np.r_[price,-price,np.zeros(self.T)]
        self.h.changeColsCost(len(q),self.idx,q);self.h.run()
        if self.h.getModelStatus()!=highspy.HighsModelStatus.kOptimal:
            # Degenerate price updates can invalidate a warm-start basis.
            self.h.clearSolver();self.h.run()
        if self.h.getModelStatus()!=highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(self.h.modelStatusToString(self.h.getModelStatus()))
        x=np.asarray(self.h.getSolution().col_value)
        return x,self.load+x[:self.T]-x[self.T:2*self.T],float(self.cost@x)


class HouseholdQP:
    def __init__(self,problem,i,rho,tol=1e-6):
        self.T=problem.T;self.cost=problem.cost;self.load=problem.loads[i].copy()
        self.G=problem.G;self.rho=rho
        A,b,upper=problem.local(i)
        A=sp.vstack([A,sp.eye(3*self.T)],format='csc')
        self.solver=osqp.OSQP()
        self.solver.setup(P=sp.triu(rho*(self.G.T@self.G),format='csc'),q=self.cost,
                          A=A,l=np.r_[b,np.zeros(3*self.T)],u=np.r_[b,upper],
                          verbose=False,eps_abs=tol,eps_rel=0,max_iter=100000,
                          polishing=True,adaptive_rho=True)

    def solve(self,z,u):
        self.solver.update(q=self.cost+self.rho*np.asarray(self.G.T@(self.load-z+u)).ravel())
        res=self.solver.solve(raise_error=False)
        if res.info.status_val!=1:
            raise RuntimeError('Household QP: '+res.info.status)
        x=res.x
        return x,self.load+x[:self.T]-x[self.T:2*self.T],float(self.cost@x)


class HouseholdLogQP:
    """Private logarithmic-savings proximal oracle for fair sharing ADMM."""
    def __init__(self,problem,i,rho):
        import cvxpy as cp
        self.cp=cp;self.T=problem.T;self.cost=problem.cost;self.load=problem.loads[i].copy()
        A,b,ub=problem.local(i);G=problem.G
        # Fixed positive scale improves numerical conditioning without changing argmin.
        optimum=HouseholdLP(problem,i).solve(np.zeros(self.T))[2]
        if optimum>=-1e-10:raise ValueError('Household has no strictly positive attainable saving')
        self.x=cp.Variable(3*self.T);self.target=cp.Parameter(self.T)
        saving=-self.cost@self.x
        self.problem=cp.Problem(cp.Minimize(-cp.log(saving/(-optimum))+
                                rho/2*cp.sum_squares(G@self.x+self.load-self.target)),
                                [A@self.x==b,self.x>=0,self.x<=ub])

    def solve(self,z,u):
        self.target.value=z-u
        self.problem.solve(solver='CLARABEL',warm_start=True,tol_gap_abs=1e-6,tol_gap_rel=1e-6,tol_feas=1e-7,max_iter=300,max_step_fraction=.8)
        if self.problem.status!='optimal':
            self.problem.solve(solver='SCS',eps=1e-6,max_iters=30000,warm_start=True)
        if self.problem.status!='optimal':raise RuntimeError('Log proximal solve: '+self.problem.status)
        x=self.x.value
        return x,self.load+x[:self.T]-x[self.T:2*self.T],float(self.cost@x)
