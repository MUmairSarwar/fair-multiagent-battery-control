import numpy as np

from fairbattery.metrics import jain_index


def test_jain_equal_is_one():
    assert jain_index(np.array([2.0, 2.0, 2.0])) == 1.0


def test_jain_one_winner():
    assert np.isclose(jain_index(np.array([5.0, 0.0, 0.0, 0.0])), 0.25)


def test_jain_clips_negative():
    assert np.isclose(jain_index(np.array([1.0, -3.0])), 0.5)
