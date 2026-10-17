import numpy as np
import pytest

from fairbattery import data
from fairbattery.env import BatteryCommunity, EnvConfig


@pytest.fixture(scope="session")
def days():
    return data.load_days()


@pytest.fixture
def env(days):
    _, price, h0 = days
    return BatteryCommunity(EnvConfig(), price, h0)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
