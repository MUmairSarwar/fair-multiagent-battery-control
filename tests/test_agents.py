import numpy as np

from fairbattery.experiment import load_config, train_and_eval


def small_cfg():
    cfg = load_config()
    cfg["train"]["iql"]["iterations"] = 5
    cfg["train"]["ppo"]["iterations"] = 2
    cfg["train"]["batch_days"] = 4
    return cfg


def test_all_agents_train_and_evaluate():
    for algo in ["iql", "ippo", "mappo"]:
        m = train_and_eval(small_cfg(), algo, seed=0)["metrics"]
        assert m["n_days"] == 51
        assert np.isfinite(m["net_benefit_eur"])


def test_same_seed_same_result():
    a = train_and_eval(small_cfg(), "ippo", seed=3)["metrics"]
    b = train_and_eval(small_cfg(), "ippo", seed=3)["metrics"]
    assert a["net_benefit_eur"] == b["net_benefit_eur"]


def test_communication_variant_runs():
    m = train_and_eval(small_cfg(), "iql", seed=0, overrides={"communication": True})["metrics"]
    assert np.isfinite(m["saving_eur"])
