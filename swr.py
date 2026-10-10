import numpy as np
import pandas as pd

CLEAN_PATH = "data/shiller_real_returns.csv"


def simulate_path(returns, initial_wealth=1_000_000, withdrawal=40_000):
    """Simulate one retirement.

    Each year: apply that year's return first, then withdraw.
    Returns an array of end-of-year wealth, one value per year.
    If wealth reaches zero, it stays at zero (the retiree is broke).
    """
    wealth = initial_wealth
    path = []
    for r in returns:
        wealth = wealth * (1 + r) - withdrawal
        if wealth <= 0:
            wealth = 0.0
        path.append(wealth)
    return np.array(path)


def load_data(path=CLEAN_PATH):
    """Load the clean Shiller dataset made by build_dataset.py."""
    return pd.read_csv(path)


def _summarize_path(path, label):
    """Turn one simulated path into one table row."""
    failed = bool((path == 0).any())
    return {
        "label": label,
        "end_wealth": float(path[-1]),
        "failed": failed,
        "failure_year": int(np.argmax(path == 0)) + 1 if failed else None,
    }


def historical_backtest(df, horizon=30, initial_wealth=1_000_000, withdrawal=40_000):
    """Run every possible `horizon`-year window in the historical data."""
    returns = df["real_return"].to_numpy()
    years = df["year"].to_numpy()

    rows = []
    for i in range(len(returns) - horizon + 1):
        path = simulate_path(returns[i:i + horizon], initial_wealth, withdrawal)
        row = _summarize_path(path, int(years[i]))
        row["start_year"] = row.pop("label")
        rows.append(row)
    return pd.DataFrame(rows)


def iid_normal(mean, std, horizon=30, n_sims=10_000, seed=42,
               initial_wealth=1_000_000, withdrawal=40_000):
    """Each year drawn independently from a normal distribution."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sims):
        returns = rng.normal(mean, std, size=horizon)
        rows.append(_summarize_path(simulate_path(returns, initial_wealth, withdrawal), i))
    return pd.DataFrame(rows)


def iid_bootstrap(returns, horizon=30, n_sims=10_000, seed=42,
                  initial_wealth=1_000_000, withdrawal=40_000):
    """Each year drawn independently, with replacement, from the historical returns."""
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns)
    rows = []
    for i in range(n_sims):
        sample = rng.choice(returns, size=horizon, replace=True)
        rows.append(_summarize_path(simulate_path(sample, initial_wealth, withdrawal), i))
    return pd.DataFrame(rows)


def block_bootstrap(returns, block_size, horizon=30, n_sims=10_000, seed=42,
                    initial_wealth=1_000_000, withdrawal=40_000):
    """Random blocks of consecutive historical years, joined and cut to `horizon` years.

    Order inside a block is kept. Blocks do not wrap around the end of the sample.
    """
    rng = np.random.default_rng(seed)
    returns = np.asarray(returns)
    n_starts = len(returns) - block_size + 1   # last start where a full block still fits
    rows = []
    for i in range(n_sims):
        seq = []
        while len(seq) < horizon:
            start = rng.integers(0, n_starts)
            seq.extend(returns[start:start + block_size])
        sample = seq[:horizon]
        rows.append(_summarize_path(simulate_path(sample, initial_wealth, withdrawal), i))
    return pd.DataFrame(rows)


def success_rate(results):
    """Share of paths that did not fail."""
    return 100 * (~results["failed"]).mean()


if __name__ == "__main__":
    # Hand test: 1,000,000 -> 1,060,000 -> 490,000 -> 450,000
    print(simulate_path([0.10, -0.50, 0.0]))

    df = load_data()
    returns = df["real_return"].to_numpy()
    mean = df["real_return"].mean()
    std = df["real_return"].std()

    hist = historical_backtest(df)
    print("historical: {} windows, {:.2f}%".format(len(hist), success_rate(hist)))
    print("failed start years:", hist.loc[hist["failed"], "start_year"].tolist())

    print("iid normal:    {:.2f}%".format(success_rate(iid_normal(mean, std))))
    print("iid bootstrap: {:.2f}%".format(success_rate(iid_bootstrap(returns))))
    for b in [3, 5, 10, 15, 20]:
        print("block {:>2} yrs: {:.2f}%".format(b, success_rate(block_bootstrap(returns, b))))