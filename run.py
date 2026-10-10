"""Run the whole study and save the figures to results/.

Run build_dataset.py once first, then run this file.
All settings live in the block below; nothing else hardcodes them.
"""
import os

from swr import (load_data, historical_backtest, iid_normal, iid_bootstrap,
                 block_bootstrap, success_rate)
import plots

# ---------------- settings ----------------
HORIZON = 30
INITIAL_WEALTH = 1_000_000
WITHDRAWAL_RATE = 0.04
N_SIMS = 10_000
SEED = 42
BLOCK_SIZES = [3, 5, 10, 15, 20]
RESULTS_DIR = "results"
# ------------------------------------------

WITHDRAWAL = INITIAL_WEALTH * WITHDRAWAL_RATE   # $40,000 a year, same real amount every year

os.makedirs(RESULTS_DIR, exist_ok=True)

df = load_data()
returns = df["real_return"].to_numpy()
mean, std = df["real_return"].mean(), df["real_return"].std()
common = dict(horizon=HORIZON, initial_wealth=INITIAL_WEALTH, withdrawal=WITHDRAWAL)
sim = dict(n_sims=N_SIMS, seed=SEED, **common)

hist = historical_backtest(df, **common)
normal = iid_normal(mean, std, **sim)
boot = iid_bootstrap(returns, **sim)
blocks = {b: block_bootstrap(returns, b, **sim) for b in BLOCK_SIZES}

# one row per method: (label, success rate as a fraction, number of paths, kind)
rows = [
    ("Historical\nwindows", success_rate(hist) / 100, len(hist), "hist"),
    ("i.i.d.\nnormal", success_rate(normal) / 100, len(normal), "iid"),
    ("i.i.d.\nbootstrap", success_rate(boot) / 100, len(boot), "iid"),
]
for b, res in blocks.items():
    rows.append((f"Block\n{b} yrs", success_rate(res) / 100, len(res), "block"))

print("{:<18}{:>8}{:>10}".format("method", "paths", "success"))
for label, rate, n, _ in rows:
    print("{:<18}{:>8,}{:>9.2f}%".format(label.replace("\n", " "), n, 100 * rate))

plots.fig1_success_rates(rows, f"{RESULTS_DIR}/fig1_success_rates.png", HORIZON)
plots.fig2_cohorts(hist, f"{RESULTS_DIR}/fig2_cohorts.png", HORIZON)
plots.fig3_failure_timing(hist, normal, f"{RESULTS_DIR}/fig3_failure_timing.png", HORIZON)
print("figures saved to", RESULTS_DIR)