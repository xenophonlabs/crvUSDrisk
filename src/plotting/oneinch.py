"""
Plotting functions for 1inch quotes 
and generated price impact curves.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..modules import ExternalMarket
from .utils import save
from ..data_transfer_objects import TokenDTO

S = 5


def plot_quotes(
    df: pd.DataFrame, in_token: TokenDTO, out_token: TokenDTO, fn: str | None = None
) -> plt.Figure:
    """Plot 1inch quotes for a given token pair."""
    f, ax = plt.subplots(figsize=(10, 5))

    scatter = ax.scatter(
        df["in_amount"] / 10**in_token.decimals,
        df["price"],
        s=S,
        c=df["timestamp"],
        cmap="viridis",
    )

    ax.set_xscale("log")
    ax.set_title(f"Prices for Swapping {in_token.symbol} into {out_token.symbol}")
    ax.set_ylabel("Exchange Rate (out/in)")
    ax.set_xlabel(f"Trade Size ({in_token.symbol})")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_yticklabels(
        pd.to_datetime(cbar.get_ticks(), unit="s").strftime("%d %b %Y")
    )
    cbar.set_label("Date")

    return save(f, fn)


# pylint: disable=too-many-arguments, too-many-locals
def plot_regression(
    df: pd.DataFrame,
    i: int,
    j: int,
    market: ExternalMarket,
    fn: str | None = None,
    scale: str = "log",
    xlim: float | None = None,
) -> plt.Figure:
    """
    Plot price impact from 1inch quotes against
    predicted price impact from market model.
    """
    in_token = market.coins[i]
    out_token = market.coins[j]

    x = np.geomspace(df["in_amount"].min(), df["in_amount"].max(), 100)
    y = market.price_impact_many(i, j, x) * 100

    f, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        df["in_amount"] / 10**in_token.decimals,
        df["price_impact"] * 100,
        c=df["timestamp"],
        s=S,
        label="1inch Quotes",
    )
    ax.plot(x / 10**in_token.decimals, y, label="Prediction", c="indianred", lw=2)
    ax.set_xscale(scale)
    ax.legend()
    ax.set_xlabel(f"Amount in ({in_token.symbol})")
    ax.set_ylabel("Price Impact %")
    ax.set_title(f"{in_token.symbol} -> {out_token.symbol} Price Impact")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_yticklabels(
        pd.to_datetime(cbar.get_ticks(), unit="s").strftime("%d %b %Y")
    )
    cbar.set_label("Date")

    if xlim:
        ax.set_xlim(0, xlim)
        ax.set_ylim(
            0,
            df[df["in_amount"] < xlim * 10**in_token.decimals]["price_impact"].max()
            * 100,
        )

    return save(f, fn)


# pylint: disable=too-many-arguments, too-many-locals
def plot_predictions(
    df: pd.DataFrame,
    i: int,
    j: int,
    market: ExternalMarket,
    fn: str | None = None,
    scale: str = "log",
    xlim: float | None = None,
) -> plt.Figure:
    """
    Plot amount in vs amount out from 1inch quotes against
    predicted amount in vs amount out from market model.
    """
    in_token = market.coins[i]
    out_token = market.coins[j]

    x = np.geomspace(df["in_amount"].min(), df["in_amount"].max(), 100)
    y = np.array([market.trade(i, j, x) for x in x])

    f, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        df["in_amount"] / 10**in_token.decimals,
        df["out_amount"] / 10**out_token.decimals,
        c=df["timestamp"],
        s=S,
        label="1inch Quotes",
    )
    ax.plot(
        x / 10**in_token.decimals,
        y / 10**out_token.decimals,
        label="Prediction",
        c="indianred",
        lw=1,
    )
    ax.set_xscale(scale)
    ax.legend()
    ax.set_xlabel(f"Amount in ({in_token.symbol})")
    ax.set_ylabel(f"Amount out ({out_token.symbol})")
    ax.set_title(f"{in_token.symbol} -> {out_token.symbol} Quotes")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.ax.set_yticklabels(
        pd.to_datetime(cbar.get_ticks(), unit="s").strftime("%d %b %Y")
    )
    cbar.set_label("Date")

    if xlim:
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, df[df["in_amount"] < xlim]["out_amount"].max())

    return save(f, fn)
