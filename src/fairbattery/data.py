"""Build the model inputs: 2025 DE-LU day-ahead prices and the BDEW H0 profile.

Both series are put on the same 15-minute grid in local German time
(Europe/Berlin). One episode in the environment is one local calendar day with
96 quarter-hours, so the two days with a clock change (23 and 25 hours) are
left out. That leaves 363 days.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PRICE_FILE = REPO_ROOT / "data" / "raw" / "energy_charts_price_DE-LU_2025.json"
PROCESSED_FILE = REPO_ROOT / "data" / "processed" / "inputs_2025.csv"

STEPS_PER_DAY = 96
DT_HOURS = 0.25

# Public holidays in Hesse in 2025 (national holidays plus Corpus Christi).
# The H0 profile treats holidays like Sundays.
HOLIDAYS_HESSE_2025 = [
    dt.date(2025, 1, 1),
    dt.date(2025, 4, 18),
    dt.date(2025, 4, 21),
    dt.date(2025, 5, 1),
    dt.date(2025, 5, 29),
    dt.date(2025, 6, 9),
    dt.date(2025, 6, 19),
    dt.date(2025, 10, 3),
    dt.date(2025, 12, 25),
    dt.date(2025, 12, 26),
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_price_quarter_hours(path: Path = RAW_PRICE_FILE) -> pd.Series:
    """Return 2025 prices (EUR/MWh) on a 15-minute UTC grid.

    Until 30 Sep 2025 the day-ahead market cleared hourly; from 1 Oct 2025 it
    clears every 15 minutes. An hourly price is valid for all four of its
    quarter-hours, so it is repeated (step function), not interpolated.
    """
    raw = json.loads(path.read_text())
    t = np.asarray(raw["unix_seconds"], dtype=np.int64)
    v = np.asarray(raw["price"], dtype=float)
    start = int(pd.Timestamp("2025-01-01", tz="Europe/Berlin").timestamp())
    end = int(pd.Timestamp("2026-01-01", tz="Europe/Berlin").timestamp())
    grid = np.arange(start, end, 900, dtype=np.int64)
    idx = np.searchsorted(t, grid, side="right") - 1
    if (idx < 0).any():
        raise ValueError("price data starts after 2025-01-01")
    gap = grid - t[idx]
    if gap.max() >= 3600:
        raise ValueError("gap of an hour or more in the price data")
    index = pd.to_datetime(grid, unit="s", utc=True)
    return pd.Series(v[idx], index=index, name="price_eur_mwh")


def h0_profile_local(year: int = 2025) -> pd.Series:
    """BDEW H0 profile with the dynamisation function, from demandlib.

    Returned as the share of the annual energy used in each quarter-hour
    (sums to 1), indexed by local wall-clock time without DST.
    """
    from demandlib import bdew

    holidays = {d: "holiday" for d in HOLIDAYS_HESSE_2025}
    slp = bdew.ElecSlp(year, holidays=holidays)
    h0 = slp.get_profiles("h0_dyn")["h0_dyn"]
    return h0 / h0.sum()


def build_inputs() -> pd.DataFrame:
    price = load_price_quarter_hours()
    h0 = h0_profile_local()
    local = price.index.tz_convert("Europe/Berlin")
    wall_clock = local.tz_localize(None)
    h0_values = h0.reindex(wall_clock).to_numpy()
    if np.isnan(h0_values).any():
        raise ValueError("H0 profile does not cover every quarter-hour")
    h0_values = h0_values / h0_values.sum()
    df = pd.DataFrame(
        {
            "timestamp_utc": price.index,
            "local_time": local,
            "local_date": wall_clock.date,
            "price_eur_mwh": price.to_numpy(),
            # kWh used in this quarter-hour by a household with 1000 kWh/year
            "h0_kwh_per_1000kwh": h0_values * 1000.0,
        }
    )
    return df


def write_processed(path: Path = PROCESSED_FILE) -> Path:
    df = build_inputs()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.6f")
    return path


def load_days(path: Path = PROCESSED_FILE):
    """Return (dates, prices [D,96] in EUR/kWh, h0 [D,96] kWh per 1000 kWh/a).

    Only days with exactly 96 quarter-hours are kept.
    """
    df = pd.read_csv(path)
    counts = df.groupby("local_date").size()
    full = counts[counts == STEPS_PER_DAY].index
    df = df[df["local_date"].isin(full)]
    dates = np.array(sorted(full))
    price = df["price_eur_mwh"].to_numpy().reshape(len(dates), STEPS_PER_DAY) / 1000.0
    h0 = df["h0_kwh_per_1000kwh"].to_numpy().reshape(len(dates), STEPS_PER_DAY)
    return dates, price, h0


def split_days(n_days: int, test_every: int = 7):
    """Hold out every 7th day as the test set, spread over the whole year.

    A purely chronological split would put all of the 15-minute price period
    (Oct-Dec) and most of winter into the test set, so the test set would
    measure a season change as much as the method.
    """
    idx = np.arange(n_days)
    test = idx[idx % test_every == test_every - 1]
    train = idx[idx % test_every != test_every - 1]
    return train, test
