"""Three figures. Each function takes results already computed by run.py and saves a PNG."""
import matplotlib

matplotlib.use("Agg")  # draw to files only, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np

GREY, ORANGE = "#333333", "#ff7f0e"
GREENS = ["#cfe8c9", "#a8d5a2", "#5cb85c", "#3c9d3c", "#2ca02c"]


def fig1_success_rates(rows, path, horizon):
    """rows: list of (label, success_rate, n_paths, kind), kind is 'hist', 'iid' or 'block'."""
    labels = [r[0] for r in rows]
    rates = np.array([r[1] for r in rows])
    cis = np.array([1.96 * np.sqrt(r[1] * (1 - r[1]) / r[2]) for r in rows])  # 95% interval

    n_block = sum(1 for r in rows if r[3] == "block")
    block_colors = iter(GREENS[-n_block:])
    colors = [GREY if r[3] == "hist" else ORANGE if r[3] == "iid" else next(block_colors)
              for r in rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, rates * 100, yerr=cis * 100, color=colors, capsize=5, ecolor="black")
    ax.axhline(rates[0] * 100, color="black", linestyle="--", alpha=0.5, label="Historical result")
    for bar, rate, ci in zip(bars, rates, cis):
        ax.text(bar.get_x() + bar.get_width() / 2, (rate + ci) * 100 + 1.2, f"{rate:.1%}",
                ha="center", fontweight="bold", fontsize=9)
    ax.set_ylim(0, 120)
    ax.set_ylabel(f"{horizon}-year success rate (%)")
    ax.set_title(f"Success Rates Under the 4% Withdrawal Rule\n"
                 f"(100% stocks, {horizon}-year retirement, real returns)")
    ax.legend(loc="upper right")
    ax.set_xlabel(f"Error bars: 95% interval. Historical n={rows[0][2]} overlapping windows; "
                  f"simulated methods n={rows[1][2]:,} paths each.", fontsize=8)
    plt.xticks(fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig2_cohorts(backtest, path, horizon):
    """One bar per retirement start year: green = money lasted, red = ran out."""
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#d62728" if f else "#2ca02c" for f in backtest["failed"]]
    ax.bar(backtest["start_year"], [1] * len(backtest), color=colors, width=1.0)
    ax.set_yticks([])
    ax.set_xlabel("Retirement start year")
    ax.set_title(f"Which {horizon}-Year Retirement Cohorts Survived? (green = lasted, red = ran out)")
    success = 1 - backtest["failed"].mean()
    ax.text(0.01, 0.92, f"Overall historical success rate: {success:.1%}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="gray", boxstyle="round,pad=0.4"))
    for start in backtest.loc[backtest["failed"], "start_year"]:
        ax.annotate(f"{start} cohort", xy=(start, 1), xytext=(start, 1.3), ha="center",
                    fontsize=8, arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_ylim(0, 1.6)
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig3_failure_timing(backtest, iid_res, path, horizon):
    """In which year of retirement does the money run out? Simulated vs historical."""
    hist = backtest.loc[backtest["failed"], "failure_year"].astype(int).to_numpy()
    sim = iid_res.loc[iid_res["failed"], "failure_year"].astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sim, bins=np.arange(0.5, horizon + 1.5, 1), density=True, color="#d62728",
            alpha=0.65, label=f"i.i.d. normal failures (n={len(sim):,})")
    for i, yr in enumerate(hist):
        label = (f"Historical failures (n={len(hist)}): year {', '.join(map(str, hist))}"
                 if i == 0 else None)
        ax.axvline(yr, color=GREY, linewidth=2.5, label=label)
    ax.set_xlabel(f"Year of retirement when the money runs out (within {horizon} years)")
    ax.set_ylabel("Share of failed paths")
    ax.set_title("When Does the Money Run Out? Simulated vs Historical Failures")
    ax.legend(loc="upper left")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)