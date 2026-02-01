import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def _apply_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def main() -> None:
    exp_dir = Path("construct_outputs")
    csv_path = exp_dir / "trial_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"trial_metrics.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("trial_metrics.csv is empty")

    df["cum_accept"] = df["accepted"].cumsum()
    df["cum_yield"] = df["cum_accept"] / df["trial_index"]

    window = 10
    df["marginal_yield"] = df["accepted"].rolling(window, min_periods=1).mean()

    _apply_style()

    # Cumulative yield
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["trial_index"], df["cum_yield"], color="#2563eb", linewidth=2, label="Cumulative yield")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Yield")
    ax.set_ylim(0, min(1.0, max(0.6, df["cum_yield"].max() + 0.05)))
    ax.legend(loc="upper right")
    last_x = df["trial_index"].iloc[-1]
    last_y = df["cum_yield"].iloc[-1]
    ax.scatter([last_x], [last_y], color="#2563eb", zorder=3)
    ax.annotate(f"final={last_y:.2f}", xy=(last_x, last_y), xytext=(8, -12),
                textcoords="offset points")
    fig.tight_layout()
    fig.savefig(exp_dir / "cumulative_yield.png", dpi=300)

    # Marginal yield
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["trial_index"], df["marginal_yield"], color="#16a34a", linewidth=2,
            label=f"Marginal yield (w={window})")
    # stop-loss reference line
    ax.axhline(0.1, color="#ef4444", linestyle="--", linewidth=1, label="Stop-loss threshold (0.1)")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Yield")
    ax.set_ylim(0, min(1.0, max(0.6, df["marginal_yield"].max() + 0.05)))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(exp_dir / "marginal_yield.png", dpi=300)


if __name__ == "__main__":
    main()
