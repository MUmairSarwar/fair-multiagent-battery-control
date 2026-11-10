"""Regenerate every Stages 1-4 figure from measured CSVs (no optimization)."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results',type=Path,default=Path(__file__).resolve().parents[1]/'results'/'distributed'/'complete')
    a=ap.parse_args();root=a.results;out=root/'figures';out.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    date='2025-01-07'
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for name in ['dual','admm']:
        d=pd.read_csv(root/f'{date}_{name}_history.csv')
        axes[0].semilogy(d.iteration,np.maximum(d.objective_gap,1e-12),label=name)
        axes[1].semilogy(d.iteration,np.maximum(d.raw_violation_kw,1e-12),label=name+' raw')
        axes[1].semilogy(d.iteration,np.maximum(d.violation_kw,1e-12),ls=':',label=name+' recovered')
    d=pd.read_csv(root/f'{date}_admm_history.csv')
    for key in ['primal_residual','dual_residual']:axes[2].semilogy(d.iteration,np.maximum(d[key],1e-12),label=key.replace('_',' '))
    for ax,title in zip(axes,['Feasible objective gap (EUR)','Transformer exceedance (kW)','ADMM residual norms']):
        ax.set_title(title);ax.set_xlabel('Iteration');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('Convergence on 7 January 2025; logarithmic axes floor values at 1e-12')
    fig.savefig(out/'convergence.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for step in [.01,.03,.1]:
        d=pd.read_csv(root/f'dual_step_{step}_history.csv');axes[0].semilogy(d.iteration,np.maximum(d.objective_gap,1e-12),label=f'step {step}')
    d=pd.read_csv(root/'dual_constant_0.03_history.csv');axes[0].semilogy(d.iteration,np.maximum(d.objective_gap,1e-12),ls='--',label='constant 0.03')
    for rho in [.01,.1,1.]:
        d=pd.read_csv(root/f'admm_rho_{rho}_history.csv');axes[1].semilogy(d.iteration,np.maximum(d.objective_gap,1e-12),label=f'rho {rho}')
    for ax,title in zip(axes,['Dual step-size sensitivity','ADMM penalty sensitivity']):
        ax.set_title(title);ax.set_xlabel('Iteration');ax.set_ylabel('Feasible objective gap (EUR)');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'sensitivity.png',dpi=160);plt.close(fig)
    d=pd.read_csv(root/'fairness.csv');fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    keys=['jain','min_saving_per_kwh','price_of_fairness'];titles=["Jain's index",'Minimum savings (EUR/kWh battery)','Price of fairness (%)']
    for ax,key,title in zip(axes,keys,titles):
        for j,kind in enumerate(['efficient','proportional','max_min']):
            sub=d[d.fairness==kind];vals=sub[key].to_numpy()*(100 if key=='price_of_fairness' else 1)
            ax.bar(np.arange(len(sub))+(j-1)*.25,vals,width=.25,label=kind)
        ax.set_xticks(range(len(sub)),sub.date.str[5:],rotation=20);ax.set_title(title)
    axes[0].set_ylim(0,1.05);axes[-1].legend(fontsize=8)
    fig.suptitle('Fairness of penalized daily savings; not a guarantee of every metric improving')
    fig.savefig(out/'fairness.png',dpi=160);plt.close(fig)
    if (root/'scaling.csv').exists():
        d=pd.read_csv(root/'scaling.csv');fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        for alg in ['central_lp','dual','admm']:
            sub=d[d.algorithm==alg].sort_values('n');axes[0].loglog(sub.n,sub.runtime_seconds,'o-',label=alg)
            if alg!='central_lp':
                axes[1].semilogx(sub.n,sub.iterations,'o-',label=alg)
                axes[2].loglog(sub.n,sub.messages_per_agent,'o-',label=alg)
                for _,row in sub.iterrows():
                    if not row.converged:axes[1].annotate('limit reached',(row.n,row.iterations),fontsize=8)
        for ax,title in zip(axes,['Measured serial runtime (s)','Iterations','Messages per agent']):
            ax.set_xlabel('Households');ax.set_title(title);ax.legend();ax.grid(alpha=.2)
        fig.suptitle('Scaling: fixed 96-interval horizon; transformer capacity = 11 N / 6 kW')
        fig.savefig(out/'scaling.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
