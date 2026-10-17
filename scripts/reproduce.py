"""Everything in one command: data -> training and evaluation -> tables and figures.

    python scripts/reproduce.py            # about 25 minutes on 2 CPU cores
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(*args):
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


if __name__ == "__main__":
    workers = sys.argv[1] if len(sys.argv) > 1 else "2"
    run(str(HERE / "prepare_data.py"))
    run(str(HERE / "run_experiments.py"), "--workers", workers)
    run(str(HERE / "make_report.py"))
