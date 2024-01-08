"""Plotting functions for simulations."""
import pandas as pd
import matplotlib.pyplot as plt
from .utils import save
from ..configs import ADDRESS_TO_SYMBOL
from ..configs.tokens import COINGECKO_IDS

COINGECKO_IDS_ = {v: k for k, v in COINGECKO_IDS.items()}


def plot_prices(
    df: pd.DataFrame, df2: pd.DataFrame | None = None, fn: str | None = None
) -> plt.Figure:
    """
    Plot prices in df. Assumes that each col
    in the df is a coin.
    """
    # Get coin names
    cols = [col for col in df.columns if col != "timestamp"]

    n = len(cols)
    n, m = int(n**0.5), n // int(n**0.5) + (n % int(n**0.5) > 0)
    # now n, m are the dimensions of the grid

    f, axs = plt.subplots(n, m, figsize=(15, 15))

    for i in range(n):
        for j in range(m):
            col = cols.pop()
            if axs.ndim == 1:
                ax = axs[j]
            else:
                ax = axs[i, j]
            ax.plot(df.index, df[col], lw=1, c="royalblue", label="Real")
            title = col
            if "0x" in title and title in ADDRESS_TO_SYMBOL:
                title = ADDRESS_TO_SYMBOL[title]
            ax.set_title(f"{title} Prices")
            ax.set_ylabel("Price (USD)")
            ax.tick_params(axis="x", rotation=45)

            if df2 is not None:
                ax.plot(
                    df2.index,
                    df2[COINGECKO_IDS_[col]],
                    lw=1,
                    c="indianred",
                    label="Simulated",
                )
                ax.legend()

    return save(f, fn)


def plot_jumps(df, recovery_threshold, fn=None):
    """Plot prices with jumps."""
    f, ax = plt.subplots(figsize=(8, 5))

    ax.plot(df["weighted_avg"], color="royalblue", lw=1, label="WAVG Price")
    ax.plot(
        df["recovery_mean"] * (1 + recovery_threshold),
        color="black",
        linestyle="--",
        lw=1,
        label="Recovery Threshold",
    )
    ax.plot(
        df["recovery_mean"] * (1 - recovery_threshold),
        color="black",
        linestyle="--",
        lw=1,
    )
    ax.set_ylabel("ETH Price (USD)")
    ax.set_title("Detecting ETH Jumps")

    for i, row in df.iterrows():
        left_edge = i - pd.to_timedelta(0.5, unit="D")
        right_edge = i + pd.to_timedelta(0.5, unit="D")
        if row["jump_state"]:
            ax.axvspan(left_edge, right_edge, color="indianred")

    ax.axvspan(None, None, color="indianred", label="Jump")
    ax.tick_params(axis="x", rotation=45)

    f.legend(loc="upper center", bbox_to_anchor=(0.5, 0), ncol=3)

    return save(f, fn)
