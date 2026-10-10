import numpy as np
import pandas as pd

from swr import (simulate_path, _summarize_path, historical_backtest,
                 block_bootstrap, load_data)


# ---------- the engine ----------

def test_hand_example():
    # 1,000,000 -> 1,060,000 -> 490,000 -> 450,000
    path = simulate_path([0.10, -0.50, 0.0])
    np.testing.assert_allclose(path, [1_060_000, 490_000, 450_000])


def test_return_is_applied_before_withdrawal():
    # return first: 1,000,000 * 1.10 - 40,000 = 1,060,000
    # (withdrawing first would give 1,056,000)
    path = simulate_path([0.10])
    assert path[0] == 1_060_000


def test_zero_returns_lose_one_withdrawal_per_year():
    path = simulate_path([0.0, 0.0, 0.0])
    np.testing.assert_allclose(path, [960_000, 920_000, 880_000])


def test_wealth_stays_at_zero_after_ruin():
    # 1,000,000 * 0.1 - 40,000 = 60,000, then 60,000 * 0.1 - 40,000 < 0 -> ruin
    path = simulate_path([-0.9, -0.9, 0.5])
    np.testing.assert_allclose(path, [60_000, 0, 0])


def test_balance_of_exactly_zero_counts_as_failure():
    path = simulate_path([0.0], initial_wealth=40_000, withdrawal=40_000)
    assert path[-1] == 0
    assert _summarize_path(path, "x")["failed"] is True


def test_path_has_one_value_per_year():
    assert len(simulate_path([0.05] * 30)) == 30


# ---------- failure bookkeeping ----------

def test_failure_year_is_first_year_at_zero():
    row = _summarize_path(np.array([60_000.0, 0.0, 0.0]), "x")
    assert row["failed"] is True
    assert row["failure_year"] == 2


def test_surviving_path_has_no_failure_year():
    row = _summarize_path(np.array([900.0, 800.0]), "x")
    assert row["failed"] is False
    assert row["failure_year"] is None


# ---------- backtest and simulation ----------

def test_backtest_window_count():
    # 40 years of data, 30-year windows -> 40 - 30 + 1 = 11 windows
    df = pd.DataFrame({"year": range(1900, 1940), "real_return": [0.05] * 40})
    assert len(historical_backtest(df)) == 11


def test_backtest_strong_returns_never_fail():
    df = pd.DataFrame({"year": range(1900, 1940), "real_return": [0.10] * 40})
    assert not historical_backtest(df)["failed"].any()


def test_block_bootstrap_same_seed_gives_same_results():
    returns = np.random.default_rng(0).normal(0.08, 0.17, 50)
    a = block_bootstrap(returns, 5, n_sims=50, seed=1)
    b = block_bootstrap(returns, 5, n_sims=50, seed=1)
    assert len(a) == 50
    pd.testing.assert_frame_equal(a, b)


# ---------- the real data ----------

def test_dataset_matches_known_values():
    df = load_data()
    assert len(df) == 142
    assert df["year"].min() == 1871 and df["year"].max() == 2012
    assert round(df["real_return"].mean(), 4) == 0.0804
    assert round(df["real_return"].std(), 4) == 0.1772