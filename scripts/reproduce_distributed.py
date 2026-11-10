"""Reproduce distributed benchmarks and all associated figures."""
import argparse,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--include-stage0',action='store_true')
a=ap.parse_args()
def run(script,*args):
    subprocess.run([sys.executable,str(root/'scripts'/script),*args],cwd=root,check=True)
if a.include_stage0:run('audit_stage0.py','--include-milp')
run('distributed_experiments.py')
run('fair_distributed_experiment.py')
run('plot_distributed.py')
