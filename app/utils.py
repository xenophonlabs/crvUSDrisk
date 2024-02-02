"""Provides utility functions for the application."""
from typing import List
import os
import base64
import pickle
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html
from dash.development.base_component import Component
import pandas as pd
from src.data_transfer_objects import TokenDTO
from src.sim.results import MonteCarloResults
from src.logging import get_logger
from src.modules import ExternalMarket


logger = get_logger(__name__)

RESULTS_DIR = os.path.join(os.getcwd(), "results")


def load_markdown_file(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as file:
        out = file.read()
        out = out.replace("# ", "### ")
        out = out.replace("crvUSD Risk Assumptions and Limitations", "")
        return out


def clean_metadata(metadata: dict) -> dict:
    """Clean metadata for display."""
    for llamma in metadata["template"].llammas:
        metadata_ = llamma.metadata
        del metadata_["llamma_params"]["bands_x"]
        del metadata_["llamma_params"]["bands_y"]
        for spool in metadata_["stableswap_pools_params"]:
            del spool["coins"]
        #     spool["coins"] = [c.symbol for c in spool["coins"]]
    return metadata


### Load data


def list_experiments() -> List[str]:
    """
    List the experiments available in the results directory.
    """
    return [
        d
        for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    ]


def list_param_sweeps(experiment) -> List[str]:
    """
    List the parameter sweeps in this directory.
    """
    experiment_dir = os.path.join(RESULTS_DIR, experiment)
    return [
        d
        for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d))
    ]


def list_scenarios(experiment, parameter) -> List[str]:
    """
    List the scenario.pkl files in this directory.
    """
    param_dir = os.path.join(RESULTS_DIR, experiment, parameter)
    return [
        d for d in os.listdir(param_dir) if os.path.isfile(os.path.join(param_dir, d))
    ]


def load_results(experiment: str, parameter: str, scenario: str) -> MonteCarloResults:
    """
    Load the file contents into a pkl.
    """
    fn = os.path.join(RESULTS_DIR, experiment, parameter, scenario)
    try:
        with open(fn, "rb") as f:
            output = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading {fn}: {e}")
        raise e
    return output


### Plotting

S = 5


def plot_quotes(df: pd.DataFrame, in_token: TokenDTO, out_token: TokenDTO) -> go.Figure:
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


def plot_regression(
    df: pd.DataFrame, i: int, j: int, market: ExternalMarket
) -> go.Figure:
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


def create_card(name: str, description: str, body: List[Component]) -> dbc.Card:
    """
    Create the main metric cards.
    """
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(
                    name,
                    className="card-title",
                ),
                html.P(description),
                body,
            ],
        ),
        style={"textAlign": "center"},
    )
