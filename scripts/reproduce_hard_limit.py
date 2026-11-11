"""Run all hard-limit experiments and redraw their figures."""
import argparse,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--include-lp-variants',action='store_true',help='also rerun compare_lp_variants.py with the MILP')
a=ap.parse_args()
def run(script,*args):
    subprocess.run([sys.executable,str(root/'scripts'/script),*args],cwd=root,check=True)
if a.include_lp_variants:run('compare_lp_variants.py','--include-milp')
run('hard_limit_experiments.py')
run('hard_limit_fair_admm.py')
run('plot_hard_limit.py')
