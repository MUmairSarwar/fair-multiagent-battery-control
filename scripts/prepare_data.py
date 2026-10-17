"""Rebuild data/processed/inputs_2025.csv from the raw price file and demandlib.

    python scripts/prepare_data.py
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fairbattery import data  # noqa: E402

EXPECTED_RAW_SHA256 = "1b4887590da71a42fefe0543ef6fdf11021454e3a4318d7270836514d26cf3af"


def main():
    got = data.sha256(data.RAW_PRICE_FILE)
    if got != EXPECTED_RAW_SHA256:
        raise SystemExit(f"raw price file changed: sha256 {got}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        path = data.write_processed()
    dates, price, h0 = data.load_days()
    print(f"wrote {path}")
    print(f"{len(dates)} full days, mean price {price.mean() * 1000:.2f} EUR/MWh")


if __name__ == "__main__":
    main()
