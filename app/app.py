"""
Provides a plotly Dash app to visualize simulation results.

Optionally run the simulation before creating the app.
"""
from datetime import datetime
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
from dash import (
    Dash,
    html,
    dcc,
    callback,
    Output,
    Input,
    State,
    no_update,
    callback_context,
)
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from src.sim.results import MonteCarloResults
from src.logging import get_logger
from src.plotting.utils import make_square
from src.configs import ADDRESS_TO_SYMBOL, LLAMMA_ALIASES, TOKEN_DTOs
from src.utils import get_quotes
from src.metrics.utils import entity_str
from .utils import (
    load_markdown_file,
    clean_metadata,
    load_results,
    run_sim,
    plot_quotes,
    plot_regression,
)

plt.switch_backend("Agg")

logger = get_logger(__name__)

DECIMALS = 3  # number of decimal places to round to

DBC_TABLE_KWARGS = {
    "responsive": True,
    "striped": True,
    "size": "lg",
    "hover": True,
    "color": "primary",
}

DIV_KWARGS = {
    "style": {"padding": "1%", "display": "inline-block", "width": "100%"},
}

SCROLL_DIV_KWARGS = {"style": {"maxHeight": "500px", "overflow": "scroll"}}

TAB_KWARGS = {"label_style": {"color": "black"}}

global output
output = None

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

load_figure_template("flatly")

initial_modal = dbc.Modal(
    [
        dbc.ModalHeader(html.H2("crvUSD Risk Simulator")),
        dbc.ModalBody(
            [
                html.P(
                    "crvUSD Risk is an Agent Based Model (ABM) that tests the resiliency of the crvUSD system under various market conditions."
                ),
                html.P("Please choose to run a simulation or load previous results."),
                dcc.Upload(
                    id="upload-results",
                    children=dbc.Button(
                        "Load Results", id="load-results-button", className="mr-1"
                    ),
                    style={"textAlign": "center"},
                ),
                html.Hr(),
                html.Div(
                    [
                        html.H5("Run Simulation", style={"textAlign": "center"}),
                        dbc.Form(
                            [
                                dbc.Label(
                                    "Select Scenario", html_for="select-scenario"
                                ),
                                dcc.Dropdown(
                                    id="select-scenario",
                                    options=[
                                        {"label": "Baseline", "value": "baseline"},
                                    ],
                                    value="baseline",
                                ),
                                dbc.Label("Select Markets", html_for="select-markets"),
                                dcc.Dropdown(
                                    [
                                        {"label": alias, "value": alias}
                                        for alias in LLAMMA_ALIASES.keys()
                                    ],
                                    ["wsteth"],
                                    # multi=True,  # TODO
                                    id="select-markets",
                                ),
                                dbc.Label("Number of Iterations", html_for="num-iter"),
                                dbc.Input(
                                    id="num-iter",
                                    type="number",
                                    min=1,
                                    max=10000,
                                    step=1,
                                    value=10,
                                ),
                                # Parameter changes (disabled for now)
                                # Responsive price config (disabled for now)
                                # Responsive liquidity config (disabled for now)
                                html.Br(),
                                html.Div(
                                    dbc.Button(
                                        "Run Simulation",
                                        id="run-sim-button",
                                        className="mr-1",
                                    ),
                                    style={"textAlign": "center"},
                                ),
                            ],
                        ),
                    ],
                ),
                html.Br(),
                dbc.Spinner(
                    html.Div(
                        [
                            html.Div(
                                id="loading-simulation",
                            ),
                        ]
                    ),
                    fullscreen=True,
                    color="primary",
                ),
            ]
        ),
    ],
    id="initial-modal",
    is_open=True,
)

app.layout = html.Div(
    [
        initial_modal,
        html.Div(id="main-content"),
    ]
)


@callback(
    [
        Output("initial-modal", "is_open"),
        Output("main-content", "children"),
        Output("loading-simulation", "children"),
    ],
    [
        Input("upload-results", "contents"),
        Input("run-sim-button", "n_clicks"),
    ],
    [
        State("initial-modal", "is_open"),
        State("select-scenario", "value"),
        State("select-markets", "value"),
        State("num-iter", "value"),
    ],
)
def generate_content(
    upload_contents,
    sim_clicks,
    is_open,
    scenario,
    markets,
    num_iter,
):
    """
    Generate main html content either from the
    loaded results or the simulation results.
    """
    ctx = callback_context

    if not ctx.triggered:
        return is_open, no_update, no_update
    else:
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        global output
        if trigger_id == "upload-results":
            output = load_results(upload_contents)
        elif trigger_id == "run-sim-button":
            output = run_sim(scenario, markets, num_iter)
        else:
            return is_open, no_update, no_update

    return False, _generate_content(output), no_update


def _generate_content(output: MonteCarloResults):
    metadata = clean_metadata(output.metadata)
    price_config = metadata["template"].pricepaths.config.copy()
    for k, v in price_config["params"].items():
        v.update({"start_price": price_config["curr_prices"][k]})

    aggregate_columns = [
        {"label": "Key Metrics", "value": "Key Metrics", "disabled": True},
        *[{"label": col, "value": col} for col in output.key_agg_cols],
        {"label": "Other", "value": "Other", "disabled": True},
        *[
            {"label": col, "value": col}
            for col in output.summary.columns
            if col not in output.key_agg_cols
        ],
    ]

    run = output.data[0]
    per_run_columns = [
        {"label": "Key Metrics", "value": "Key Metrics", "disabled": True},
        *[{"label": col, "value": col} for col in run.key_metrics],
        {"label": "Other", "value": "Other", "disabled": True},
        *[
            {"label": col, "value": col}
            for col in run.cols
            if col not in run.key_metrics
        ],
    ]

    assets = [
        {"label": ADDRESS_TO_SYMBOL[asset], "value": asset}
        for asset in [c.address for c in metadata["template"].coins]
    ]

    worst_depeg_agg = output.summary.loc[
        output.summary[["Peg Strength Min", "Peg Strength Max"]]
        .subtract(1)
        .abs()
        .stack()
        .idxmax()
    ]

    spools = [
        entity_str(spool, "stableswap").title()
        for spool in metadata["template"].stableswap_pools
    ]
    worst_depeg_spool = None
    worst_depeg_spool_val = None
    for spool in spools:
        val = output.summary.loc[
            output.summary[[f"{spool} Price Max", f"{spool} Price Min"]]
            .subtract(1)
            .abs()
            .stack()
            .idxmax()
        ]
        if worst_depeg_spool_val is None or val > worst_depeg_spool_val:
            worst_depeg_spool = "/".join(spool.split("_")[1:]).upper()
            worst_depeg_spool_val = val
    worst_depeg_spool, worst_depeg_spool_val

    depeg_str = html.Div(
        [
            html.H5(f"Worst Depeg in Aggregator: {worst_depeg_agg:,.2f}"),
            html.H5(
                f"Worst Depeg in StableSwap: {worst_depeg_spool_val:,.2f} in {worst_depeg_spool}"
            ),
        ]
    )

    layout = html.Div(
        [
            html.H1(
                "crvUSD Risk Simulation Results",
                style={"textAlign": "center"},
            ),
            html.Div(
                dbc.Alert(
                    [
                        html.H3("Overview", style={"textAlign": "center"}),
                        html.Div(
                            html.Ul(
                                [
                                    html.Li(
                                        [
                                            html.Span(
                                                "Scenario Name: ",
                                                style={"font-weight": "bold"},
                                            ),
                                            metadata["scenario"],
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.Span(
                                                "Number of Iterations: ",
                                                style={"font-weight": "bold"},
                                            ),
                                            metadata["num_iter"],
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.Span(
                                                f"Simulation Horizon: ",
                                                style={"font-weight": "bold"},
                                            ),
                                            metadata["num_steps"],
                                            " steps of ",
                                            metadata["freq"],
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.Span(
                                                "Markets: ",
                                                style={"font-weight": "bold"},
                                            ),
                                            str(metadata["markets"]),
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.Span(
                                                "Brief description: ",
                                                style={"font-weight": "bold"},
                                            ),
                                            str(metadata["description"]),
                                        ]
                                    ),
                                ]
                            ),
                        ),
                        html.H5("Disclaimers", style={"textAlign": "center"}),
                        html.P(
                            "Not financial advice. All assumptions and limitations documented in the INFO tab.",
                            style={"textAlign": "center"},
                        ),
                        html.Div(
                            [
                                dbc.Button(
                                    "Download Output",
                                    id="download-button",
                                    color="secondary",
                                ),
                                dcc.Download(id="download-output"),
                            ],
                            style={"textAlign": "center"},
                        ),
                    ],
                    color="primary",
                ),
                **DIV_KWARGS,
            ),
            dbc.Tabs(
                [
                    dbc.Tab(
                        html.Div(
                            [
                                html.H2("Summary", style={"textAlign": "center"}),
                                html.P(
                                    "Summarized metrics, histograms, and raw data across all model runs.",
                                    style={"textAlign": "center"},
                                ),
                                dbc.CardGroup(
                                    [
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.H4(
                                                        "Value at Risk",
                                                        className="card-title",
                                                    ),
                                                    html.P(
                                                        "Value at Risk (VaR) is the p99 maximum bad debt observed over the simulated runs. This may intuitively be interpreted as: Bad debt under the input assumptions will only ever exceed VaR 1% of the time."
                                                    ),
                                                    html.H5(
                                                        f"VaR: {output.summary['Bad Debt Max'].quantile(0.99):,.0f} crvUSD"
                                                    ),
                                                ],
                                            ),
                                            style={"textAlign": "center"},
                                        ),
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.H4(
                                                        "Liquidations at Risk",
                                                        className="card-title",
                                                    ),
                                                    html.P(
                                                        "Liquidations at Risk (LaR) is the p99 maximum collateral liquidated over the simulated runs. This may intuitively be interpreted as: Liquidated collateral under the input assumptions will only ever exceed LaR 1% of the time."
                                                    ),
                                                    html.H5(
                                                        f"LaR: {output.summary['Collateral Liquidated Max'].quantile(0.99):,.0f} {ADDRESS_TO_SYMBOL[metadata['template'].llamma.COLLATERAL_TOKEN.address]}"
                                                    ),
                                                ],
                                            ),
                                            style={"textAlign": "center"},
                                        ),
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.H4(
                                                        "Borrower Losses at Risk",
                                                        className="card-title",
                                                    ),
                                                    html.P(
                                                        "Borrower Losses at Risk (BLaR) is the p99 maximum borrower losses observed over the simulated runs. This may intuitively be interpreted as: Borrower losses under the input assumptions will only ever exceed BLaR 1% of the time."
                                                    ),
                                                    html.H5(
                                                        f"BLaR: {output.summary['Borrower Loss Max'].quantile(0.99):,.0f} crvUSD"
                                                    ),
                                                ],
                                            ),
                                            style={"textAlign": "center"},
                                        ),
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.H4(
                                                        "Worst Depeg",
                                                        className="card-title",
                                                    ),
                                                    html.P(
                                                        "The worst depeg (up or down) observed over the simulated runs for the Aggregator or any StableSwap pool."
                                                    ),
                                                    html.H5(depeg_str),
                                                ],
                                            ),
                                            style={"textAlign": "center"},
                                        ),
                                    ]
                                ),
                                html.Br(),
                                html.H4(
                                    "Metric Histograms", style={"textAlign": "center"}
                                ),
                                dbc.Row(
                                    dbc.Col(
                                        dbc.Select(
                                            options=aggregate_columns,
                                            id="aggregate-metric-dropdown",
                                            value="System Health Mean",
                                        ),
                                        width=4,
                                    ),
                                    justify="center",
                                ),
                                dcc.Graph(id="aggregate-graph"),
                                html.H4(
                                    "Aggregate Data", style={"textAlign": "center"}
                                ),
                                html.Br(),
                                html.Div(
                                    dbc.Table.from_dataframe(
                                        output.summary.reset_index(
                                            names=["Run ID"]
                                        ).round(DECIMALS),
                                        **DBC_TABLE_KWARGS,
                                        id="aggregate-data",
                                    ),
                                    **SCROLL_DIV_KWARGS,
                                ),
                            ],
                            **DIV_KWARGS,
                        ),
                        label="Summary",
                        **TAB_KWARGS,
                    ),
                    dbc.Tab(
                        html.Div(
                            [
                                html.H2(
                                    "Per Simulated Run", style={"textAlign": "center"}
                                ),
                                html.P(
                                    "Timeseries plots, metrics data, and prices for each simulated run.",
                                    style={"textAlign": "center"},
                                ),
                                html.H4(
                                    "Per Simulated Run Plots",
                                    style={"textAlign": "center"},
                                ),
                                dbc.Row(
                                    dbc.Col(
                                        dbc.Select(
                                            options=per_run_columns,
                                            id="run-metric-dropdown",
                                            value="System Health",
                                        ),
                                        width=4,
                                    ),
                                    justify="center",
                                ),
                                dcc.Graph(id="run-graph"),
                                html.H4(
                                    "Per Simulated Run Data",
                                    style={"textAlign": "center"},
                                ),
                                dbc.Row(
                                    dbc.Col(
                                        [
                                            dbc.Label("Select a run:"),
                                            dbc.Input(
                                                id="run-dropdown",
                                                type="number",
                                                min=1,
                                                max=len(output.data),
                                                step=1,
                                                value=0,
                                            ),
                                        ],
                                        width=3,
                                    ),
                                    justify="center",
                                ),
                                html.Br(),
                                html.Div(id="run-data-container", **SCROLL_DIV_KWARGS),
                                html.Br(),
                                html.H4(
                                    "Per Run Prices", style={"textAlign": "center"}
                                ),
                                dbc.Row(
                                    dbc.Col(
                                        dbc.Checkbox(
                                            label="Show All",
                                            value=True,
                                            id="price-checkbox",
                                        ),
                                        width=1,
                                    ),
                                    justify="center",
                                ),
                                dcc.Graph(id="run-prices"),
                            ],
                            **DIV_KWARGS,
                        ),
                        label="Per Run Data",
                        **TAB_KWARGS,
                    ),
                    dbc.Tab(
                        html.Div(
                            [
                                html.H2("Configuration", style={"textAlign": "center"}),
                                html.P(
                                    "Simulation configuration.",
                                    style={"textAlign": "center"},
                                ),
                                html.H4(
                                    "Price Parameters", style={"textAlign": "center"}
                                ),
                                html.P(
                                    f"Generative parameters for each stochastic process. Trained from {datetime.fromtimestamp(price_config['start'])} to {datetime.fromtimestamp(price_config['end'])}",
                                    style={"textAlign": "center"},
                                ),
                                dbc.Row(
                                    dbc.Col(
                                        html.Div(
                                            dbc.Table.from_dataframe(
                                                pd.DataFrame.from_dict(
                                                    price_config["params"],
                                                    orient="index",
                                                )
                                                .reset_index(names="Name")
                                                .round(DECIMALS),
                                                **DBC_TABLE_KWARGS,
                                            ),
                                            **SCROLL_DIV_KWARGS,
                                        ),
                                        width=6,
                                    ),
                                    justify="center",
                                ),
                                html.Br(),
                                html.H4(
                                    "Asset covariances", style={"textAlign": "center"}
                                ),
                                html.Div(
                                    dbc.Table.from_dataframe(
                                        pd.DataFrame.from_dict(
                                            price_config["cov"],
                                            orient="index",
                                        )
                                        .reset_index(names="Name")
                                        .round(DECIMALS),
                                        **DBC_TABLE_KWARGS,
                                    ),
                                    **SCROLL_DIV_KWARGS,
                                ),
                                html.Br(),
                                html.H4(
                                    "External Liquidity Curves",
                                    style={"textAlign": "center"},
                                ),
                                html.P(
                                    f"Quotes from {metadata['template'].quotes_start} to {metadata['template'].quotes_end}.",
                                    style={"textAlign": "center"},
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label(
                                                            "Select In Asset",
                                                            html_for="select-in-asset",
                                                        ),
                                                        dbc.Select(
                                                            options=assets,
                                                            id="select-in-asset",
                                                            value=assets[0]["value"],
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Label(
                                                            "Select Out Asset",
                                                            html_for="select-out-asset",
                                                        ),
                                                        dbc.Select(
                                                            options=assets,
                                                            id="select-out-asset",
                                                            value=assets[-1]["value"],
                                                        ),
                                                    ],
                                                    width=3,
                                                ),
                                            ],
                                            justify="center",
                                        ),
                                        html.Br(),
                                        dbc.Button(
                                            "Fetch and Show Liquidity Curve",
                                            id="fetch-liquidity-button",
                                            className="mr-1",
                                        ),
                                        html.Div(
                                            id="fetch-liquidity-output",
                                        ),
                                    ],
                                    style={"textAlign": "center"},
                                ),
                                html.Br(),
                                html.Br(),
                                html.H4(
                                    "Module Metadata", style={"textAlign": "center"}
                                ),
                                html.Div(
                                    dcc.Markdown(
                                        f"```json\n{json.dumps(metadata['template'].llamma.metadata, indent=4)}\n```"
                                    ),
                                    **SCROLL_DIV_KWARGS,
                                ),
                            ],
                            **DIV_KWARGS,
                        ),
                        label="Config",
                        **TAB_KWARGS,
                    ),
                    dbc.Tab(
                        html.Div(
                            [
                                html.H2(
                                    "Assumptions and Limitations",
                                    style={"textAlign": "center"},
                                ),
                                dcc.Markdown(
                                    load_markdown_file("assumptions.md"),
                                    dangerously_allow_html=True,
                                ),
                            ],
                            **DIV_KWARGS,
                        ),
                        label="Info",
                        **TAB_KWARGS,
                    ),
                ],
            ),
        ]
    )

    return layout


@callback(
    Output("download-output", "data"),
    Input("download-button", "n_clicks"),
    prevent_initial_call=True,
)
def save_output(n_clicks):
    if n_clicks:
        return dcc.send_bytes(
            pickle.dumps(output), filename=f"{output.metadata['scenario']}.pkl"
        )
    return no_update


@callback(
    Output("aggregate-graph", "figure"), Input("aggregate-metric-dropdown", "value")
)
def update_aggregate_graph(value):
    dff = output.summary[value]
    return px.histogram(dff)


@callback(Output("run-graph", "figure"), Input("run-metric-dropdown", "value"))
def update_run_graph(value):
    """
    Plot `value` for all dataframes in output.data.
    """
    fig = go.Figure()

    for run in output.data:
        dff = run.df[value]
        fig.add_trace(go.Scatter(x=dff.index, y=dff, mode="lines"))

    fig.update_layout(
        title=f"Plot of {value} for all Runs",
        xaxis_title="Date",
        yaxis_title=value,
    )

    return fig


@callback(Output("run-data-container", "children"), Input("run-dropdown", "value"))
def update_run_data_table(value: int):
    if not output:
        return no_update
    return dbc.Table.from_dataframe(
        output.data[value - 1].df.reset_index(names=["Time"]).round(DECIMALS),
        **DBC_TABLE_KWARGS,
    )


@callback(
    Output("fetch-liquidity-output", "children"),
    Input("fetch-liquidity-button", "n_clicks"),
    [
        State("select-in-asset", "value"),
        State("select-out-asset", "value"),
    ],
)
def fetch_liquidity_curves(n_clicks, in_asset, out_asset):
    if not n_clicks:
        return no_update
    else:
        start = output.metadata["template"].quotes_start
        end = output.metadata["template"].quotes_end
        in_asset_dto = TOKEN_DTOs[in_asset]
        out_asset_dto = TOKEN_DTOs[out_asset]
        pair = tuple(sorted((in_asset_dto, out_asset_dto)))
        quotes = get_quotes(
            int(start.timestamp()),
            int(end.timestamp()),
            pair,
        )
        i = pair.index(in_asset_dto)
        j = pair.index(out_asset_dto)
        market = output.metadata["template"].markets[pair]
        df = quotes.loc[in_asset_dto.address, out_asset_dto.address]
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Graph(
                                figure=plot_quotes(df, in_asset_dto, out_asset_dto)
                            ),
                        ),
                        dbc.Col(
                            dcc.Graph(figure=plot_regression(df, i, j, market)),
                        ),
                    ]
                )
            ]
        )


@callback(
    Output("run-prices", "figure"),
    Input("run-dropdown", "value"),
    Input("price-checkbox", "value"),
)
def update_run_prices(value: int, show_all: bool):
    if not output:
        return no_update
    df = output.data[value - 1].pricepaths.prices
    cols = [col for col in df.columns if col != "timestamp"]
    titles = [ADDRESS_TO_SYMBOL[col] for col in cols]
    n, m = make_square(len(cols))
    fig = make_subplots(rows=n, cols=m, start_cell="bottom-left", subplot_titles=titles)

    for i in range(n):
        for j in range(m):
            if len(cols):
                col = cols.pop(0)
                if show_all:
                    for _df in output.data:
                        prices = _df.pricepaths.prices
                        fig.add_trace(
                            go.Scatter(x=prices.index, y=prices[col], mode="lines"),
                            row=i + 1,
                            col=j + 1,
                        )
                else:
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df[col], mode="lines"),
                        row=i + 1,
                        col=j + 1,
                    )
            else:
                break

    fig.update_layout(
        title=f"Prices for run {value}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        height=1000,
        showlegend=False,
    )

    return fig


if __name__ == "__main__":
    app.run()
