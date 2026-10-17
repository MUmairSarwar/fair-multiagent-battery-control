"""Train and evaluate every algorithm, variant and seed listed in configs/.

    python scripts/run_experiments.py --workers 2
    python scripts/run_experiments.py --quick     # tiny run for CI, writes to results/quick
"""

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.simplefilter("ignore", FutureWarning)

import yaml  # noqa: E402

from fairbattery.experiment import CONFIG_DIR, RESULTS_DIR, load_config, run_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="a few iterations only, to check that everything runs")
    args = ap.parse_args()
    cfg = load_config()
    exp = yaml.safe_load((CONFIG_DIR / "experiments.yaml").read_text())
    out = RESULTS_DIR
    if args.quick:
        cfg["train"]["iql"]["iterations"] = 5
        cfg["train"]["ppo"]["iterations"] = 2
        cfg["train"]["seeds"] = [0]
        cfg["train"]["batch_days"] = 4
        out = RESULTS_DIR / "quick"
    run_all(cfg, exp, out_dir=out, workers=args.workers)


if __name__ == "__main__":
    main()
