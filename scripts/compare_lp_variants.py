"""Compare the LP variants and the exclusive MILP on the 51 test days (no RL training)."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fairbattery import data
from fairbattery.lp_variants import run_variant, sufficient_throughput_cost
from fairbattery.env import BatteryCommunity, EnvConfig
from fairbattery.metrics import baseline_cost, summarise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=data.REPO_ROOT / 'results' / 'distributed' / 'lp_variants')
    parser.add_argument('--include-milp', action='store_true', help='Also solve the exclusive-mode MILP (slower); raises if a day hits the time limit')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg_path = data.REPO_ROOT / 'configs' / 'default.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    dates, prices, h0 = data.load_days()
    _, days = data.split_days(len(dates))
    env = BatteryCommunity(EnvConfig(**cfg['env']), prices, h0)
    # Generate the full test batch once: exact legacy random-number ordering.
    env.reset(days, np.random.default_rng(cfg['eval']['load_seed']))
    base = baseline_cost(env)
    fee = sufficient_throughput_cost(env.price, env.eta)
    variants = [('original_lp', 0.0, False), ('small_fee_lp', 1e-4, False),
                ('sufficient_fee_lp', fee, False)]
    if args.include_milp:
        variants.append(('exclusive_milp', 0.0, True))
    rows, raw = [], {}
    for name, kappa, exclusive in variants:
        print(f'Solving {name} ...', flush=True)
        start = perf_counter()
        result = run_variant(env, throughput_cost=kappa, exclusive=exclusive)
        elapsed = perf_counter() - start
        row = summarise(result, env, base)
        c, d, e = (result[k] for k in ('charge_kwh', 'discharge_kwh', 'energy_kwh'))
        both = (c > 1e-6) & (d > 1e-6)
        row.update({
            'variant': name, 'throughput_cost_eur_per_kwh': kappa,
            'objective_eur': float(result['objective_eur'].sum()),
            'throughput_fee_eur': float(result['throughput_fee_eur'].sum()),
            'throughput_kwh': float((c+d).sum()),
            'simultaneous_household_steps': int(both.sum()),
            'simultaneous_clock_steps': int(both.any(axis=1).sum()),
            'simultaneous_at_negative_price': int((both & (env.price[:, None, :] < 0)).sum()),
            'max_overlap_kwh': float(np.minimum(c,d).max()),
            'runtime_seconds': elapsed,
            'max_soc_violation_kwh': float(max(0, -e.min(), (e-env.cap[None,:,None]).max())),
            'max_grid_nonneg_violation_kwh': float(max(0, -result['grid_kwh'].min())),
            'max_import_exceedance_kw': float(np.maximum(result['total_kw']-env.cfg.transformer_kw,0).max()),
            'max_mip_gap': float(np.max(result['mip_gaps'])) if exclusive else None,
        })
        np.testing.assert_allclose(row['objective_eur'],
                                   -row['net_benefit_eur'] + row['throughput_fee_eur'], atol=1e-6)
        rows.append(row)
        raw[name] = result
        np.savez_compressed(args.out / f'{name}.npz', charge_kwh=c, discharge_kwh=d,
                            energy_kwh=e, total_kw=result['total_kw'],
                            objective_eur=result['objective_eur'])
        print(json.dumps({k: row[k] for k in ('variant','net_benefit_eur',
                         'simultaneous_household_steps','runtime_seconds')}), flush=True)
    orig = rows[0]['net_benefit_eur']
    for row in rows:
        row['original_benefit_loss_eur'] = orig - row['net_benefit_eur']
        row['original_benefit_loss_pct'] = 100 * row['original_benefit_loss_eur'] / abs(orig)
    pd.DataFrame(rows).to_csv(args.out / 'comparison.csv', index=False)
    report = {
        'scope': 'LP variants and exclusive MILP; soft transformer penalty; independent daily horizons',
        'seed': cfg['eval']['load_seed'], 'overlap_tolerance_kwh': 1e-6,
        'test_dates': dates[days].tolist(), 'config': cfg,
        'input_sha256': hashlib.sha256(data.PROCESSED_FILE.read_bytes()).hexdigest(),
        'config_sha256': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
        'base_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=data.REPO_ROOT, text=True).strip(),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'scipy': scipy.__version__, 'pandas': pd.__version__},
        'results': rows,
    }
    (args.out / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    plot_results(raw, env, dates[days], args.out)


LABELS = {'original_lp': 'Original LP', 'small_fee_lp': 'Small fee', 'sufficient_fee_lp': 'Sufficient fee', 'exclusive_milp': 'Exclusive MILP'}


def plot_results(raw, env, dates, out):
    names = list(raw)
    fig, axes = plt.subplots(1, 2, figsize=(11,4.2), layout='constrained')
    counts = [np.count_nonzero((raw[n]['charge_kwh']>1e-6)&(raw[n]['discharge_kwh']>1e-6)) for n in names]
    benefits = [float(raw[n]['saving'].sum()-env.cfg.overload_penalty*raw[n]['overload_kwh'].sum()) for n in names]
    labels = [LABELS[n] for n in names]
    for ax, vals, title in zip(axes, [counts, benefits],
                             ['Household-steps charging and discharging at once','Net benefit (EUR)']):
        bars = ax.bar(labels, vals, color=['#777777','#cf8d30','#31877a','#285a91'])
        ax.bar_label(bars, fmt='%.3f' if ax is axes[1] else '%.0f', padding=3)
        ax.set_title(title); ax.tick_params(axis='x', rotation=18)
        ax.set_ylim(0, max(vals)*1.2); ax.spines[['top','right']].set_visible(False)
    fig.suptitle('51 test days, identical loads; benefit shown without the throughput fee')
    fig.savefig(out/'comparison.png', dpi=180); plt.close(fig)
    old = raw['original_lp']; both=(old['charge_kwh']>1e-6)&(old['discharge_kwh']>1e-6)
    # Pick the household/day containing the most cycling, deterministically.
    b,i = np.unravel_index(both.sum(axis=2).argmax(), both.shape[:2])
    fig, axes=plt.subplots(3,1,figsize=(10,7),sharex=True,layout='constrained')
    x=np.arange(96)*data.DT_HOURS
    axes[0].step(x,env.price[b]*1000,where='post',color='#222222'); axes[0].set_ylabel('EUR/MWh')
    comparison = 'exclusive_milp' if 'exclusive_milp' in raw else 'sufficient_fee_lp'
    for ax, name in zip(axes[1:], ['original_lp', comparison]):
        r=raw[name]
        ax.step(x,r['charge_kwh'][b,i]/data.DT_HOURS,where='post',label='Charge')
        ax.step(x,r['discharge_kwh'][b,i]/data.DT_HOURS,where='post',label='Discharge')
        ax.set_ylabel('Power (kW)'); ax.set_title(LABELS[name]); ax.legend(loc='upper right')
    axes[-1].set_xlabel('Local hour'); fig.suptitle(f'{dates[b]}, household {i+1}: original LP and corrected schedule')
    fig.savefig(out/'example_day.png',dpi=180); plt.close(fig)


if __name__ == '__main__':
    main()
