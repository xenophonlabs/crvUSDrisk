"""Plotting functions for simulations."""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import plotly.graph_objects as go
import numpy as np
from .utils import save, remove_spines_and_ticks
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


def plot_jumps(
    df: pd.Series, recovery_threshold: float, fn: str | None = None
) -> plt.Figure:
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


# pylint: disable=too-many-arguments
def plot_borrowers_2d(
    health: np.ndarray,
    collateral: np.ndarray,
    density: np.ndarray,
    xlim: tuple | None = None,
    ylim: tuple | None = None,
    num_bins: int = 50,
) -> plt.Figure:
    """
    Plot borrower distribution.
    """
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(4, 4, wspace=0.1, hspace=0.1)

    ax = fig.add_subplot(gs[1:4, 0:3])
    ax.scatter(health, collateral, c=density, s=2)
    ax.set_xlabel("Health")
    ax.set_ylabel("Log of Collateral")

    # Apply x and y limits if provided
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax_histx = fig.add_subplot(gs[0, 0:3])
    ax_histx.hist(health, bins=num_bins, density=True, color="#450e58")
    ax_histx = remove_spines_and_ticks(ax_histx, skip=["bottom"], keep_x_ticks=True)

    if xlim is not None:
        ax_histx.set_xlim(*xlim)

    ax_histy = fig.add_subplot(gs[1:4, 3])
    ax_histy.hist(
        collateral,
        bins=num_bins,
        density=True,
        orientation="horizontal",
        color="#450e58",
    )
    ax_histy = remove_spines_and_ticks(ax_histy, skip=["left"], keep_y_ticks=True)
    ax_histx.set_title("Borrower States")

    if ylim is not None:
        ax_histy.set_ylim(*ylim)

    return fig


def plot_borrowers_3d(
    health: np.ndarray, collateral_log: np.ndarray, n: np.ndarray, density: np.ndarray
) -> go.Figure:
    """
    Plot borrower distribution.
    """
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=health,
                y=collateral_log,
                z=n,
                mode="markers",
                marker={
                    "size":2,
                    "color":density,  # set color to an array/list of desired values
                    "colorscale":"Viridis",  # choose a colorscale
                    "opacity":0.1,
                },
            )
        ]
    )

    fig.update_layout(
        title="Borrower Stats",
        xaxis_title="Health",
        yaxis_title="Log Collateral",
        # zaxis_title="N",
        height=500,
        showlegend=False,
    )

    return fig
