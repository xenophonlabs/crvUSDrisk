"""Provides utility functions for the application."""
from copy import deepcopy
import base64
import pickle
from datetime import datetime
from multiprocessing import cpu_count
import numpy as np
import plotly.graph_objects as go
from src.sim import run_scenario
from src.sim.results import MonteCarloResults
from src.logging import get_logger


logger = get_logger(__name__)


def load_markdown_file(filename):
    with open(filename, "r", encoding="utf-8") as file:
        out = file.read()
        out = out.replace("# ", "### ")
        out = out.replace("crvUSD Risk Assumptions and Limitations", "")
        return out


def clean_metadata(metadata):
    """Clean metadata for display."""
    metadata = deepcopy(metadata)
    metadata_ = metadata["template"].llamma.metadata
    metadata["bands_x"] = metadata_["llamma_params"]["bands_x"].copy()
    del metadata_["llamma_params"]["bands_x"]
    metadata["bands_y"] = metadata_["llamma_params"]["bands_y"].copy()
    del metadata_["llamma_params"]["bands_y"]
    for spool in metadata_["stableswap_pools_params"]:
        spool["coins"] = [c.symbol for c in spool["coins"]]
    return metadata


def load_results(contents) -> MonteCarloResults:
    """
    Load the file contents using pickle and return the output object.
    """
    output = pickle.loads(base64.b64decode(contents.split(",")[1]))
    return output


def run_sim(scenario, markets, num_iter) -> MonteCarloResults:
    """
    Run the simulation with the input parameters and return the output object
    """
    start = datetime.now()
    output = run_scenario(scenario, markets[0], num_iter=num_iter, ncpu=cpu_count())
    end = datetime.now()
    diff = end - start
    logger.info("Done. Total runtime: %s", diff)
    return output


### Plotting

S = 5


def plot_quotes(df, in_token, out_token):
    """Plot 1inch quotes for a given token pair."""
    tickvals = np.linspace(df["timestamp"].min(), df["timestamp"].max(), num=10)
    ticktext = [datetime.utcfromtimestamp(tv).strftime("%d %b %Y") for tv in tickvals]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["in_amount"] / 10**in_token.decimals,
            y=df["price"],
            mode="markers",
            marker=dict(
                color=df["timestamp"],
                colorscale="Viridis",
                size=S,
                colorbar=dict(
                    tickvals=tickvals,
                    ticktext=ticktext,
                ),
            ),
        ),
    )

    fig.update_xaxes(type="log", title_text=f"Amount in ({in_token.symbol})")

    # Set y-axis label and title
    fig.update_yaxes(title_text="Price (out/in)")
    fig.update_layout(
        title=f"{in_token.symbol} -> {out_token.symbol} Quotes", showlegend=False
    )

    return fig


def plot_regression(df, i, j, market):
    """
    Plot price impact from 1inch quotes against
    predicted price impact from market model.
    """
    in_token = market.coins[i]
    out_token = market.coins[j]

    x = np.geomspace(df["in_amount"].min(), df["in_amount"].max(), 100)
    y = market.price_impact_many(i, j, x) * 100

    fig = go.Figure()

    tickvals = np.linspace(df["timestamp"].min(), df["timestamp"].max(), num=10)
    ticktext = [datetime.utcfromtimestamp(tv).strftime("%d %b %Y") for tv in tickvals]

    fig.add_trace(
        go.Scatter(
            x=df["in_amount"] / 10**in_token.decimals,
            y=df["price_impact"] * 100,
            mode="markers",
            marker=dict(
                color=df["timestamp"],
                colorscale="Viridis",
                size=S,
                colorbar=dict(
                    tickvals=tickvals,
                    ticktext=ticktext,
                ),
            ),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=x / 10**in_token.decimals,
            y=y,
            mode="lines",
            line=dict(color="indianred"),
        ),
    )

    fig.update_xaxes(type="log", title_text=f"Amount in ({in_token.symbol})")

    # Set y-axis label and title
    fig.update_yaxes(title_text="Price Impact %")
    fig.update_layout(
        title=f"{in_token.symbol} -> {out_token.symbol} Price Impact", showlegend=False
    )

    return fig
