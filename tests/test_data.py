import numpy as np

from fairbattery import data


def test_days_shape(days):
    dates, price, h0 = days
    assert len(dates) == 363  # 365 minus the two clock-change days
    assert price.shape == (363, 96) and h0.shape == (363, 96)
    assert not np.isnan(price).any() and not np.isnan(h0).any()


def test_first_price_matches_smard(days):
    _, price, _ = days
    # first hour of 2025 on SMARD: 2.16 EUR/MWh, repeated over 4 quarter-hours
    assert np.allclose(price[0, :4] * 1000, 2.16)


def test_h0_is_scaled_to_1000_kwh(days):
    _, _, h0 = days
    # 363 of 365 days, so a little under 1000 kWh
    assert 985 < h0.sum() < 1000


def test_split_has_no_overlap():
    train, test = data.split_days(363)
    assert len(set(train) & set(test)) == 0
    assert len(train) + len(test) == 363
