"""
Provides a plotly Dash app to visualize simulation results.

Optionally run the simulation before creating the app.
"""
import time
from datetime import datetime
import pickle
import json
import base64
from multiprocessing import cpu_count
import pandas as pd
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
from src.sim import run_scenario
from src.sim.results import MonteCarloResults
from src.logging import get_logger
from src.plotting.utils import make_square
from src.configs import ADDRESS_TO_SYMBOL, LLAMMA_ALIASES
from .utils import load_markdown_file, clean_metadata

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
SCROLL_DIV_KWARGS_50 = {
    "style": {"maxHeight": "500px", "overflow": "scroll", "width": "50%"}
}

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
                                    multi=True,
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


@app.callback(
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
                            dbc.Row(
                                [
                                    dbc.Col(
                                        html.Ul(
                                            [
                                                html.Li(
                                                    [
                                                        html.Span(
                                                            "Scenario Name: ",
                                                            style={
                                                                "font-weight": "bold"
                                                            },
                                                        ),
                                                        metadata["scenario"],
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.Span(
                                                            "Number of Iterations: ",
                                                            style={
                                                                "font-weight": "bold"
                                                            },
                                                        ),
                                                        metadata["num_iter"],
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.Span(
                                                            f"Simulation Horizon: ",
                                                            style={
                                                                "font-weight": "bold"
                                                            },
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
                                                            style={
                                                                "font-weight": "bold"
                                                            },
                                                        ),
                                                        str(metadata["markets"]),
                                                    ]
                                                ),
                                                html.Li(
                                                    [
                                                        html.Span(
                                                            "Brief description: ",
                                                            style={
                                                                "font-weight": "bold"
                                                            },
                                                        ),
                                                        str(metadata["description"]),
                                                    ]
                                                ),
                                            ]
                                        ),
                                        width=9,
                                    ),
                                    dbc.Col(
                                        [
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
                                            )
                                        ],
                                        width=3,
                                    ),
                                ],
                            )
                        ),
                        html.H5("Disclaimers", style={"textAlign": "center"}),
                        html.P(
                            "Not financial advice. All assumptions and limitations documented in the INFO tab.",
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
                                html.H2(
                                    "Aggregated Data", style={"textAlign": "center"}
                                ),
                                html.P(
                                    "Aggregated histograms and raw data across all model runs.",
                                    style={"textAlign": "center"},
                                ),
                                html.H4("Metric Histograms"),
                                html.Div(
                                    dbc.Select(
                                        options=aggregate_columns,
                                        id="aggregate-metric-dropdown",
                                        value="System Health Mean",
                                    ),
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(id="aggregate-graph"),
                                html.H4("Aggregate Data"),
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
                        label="Aggregate Data",
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
                                html.H4("Per Simulated Run Plots"),
                                html.Div(
                                    dbc.Select(
                                        options=per_run_columns,
                                        id="run-metric-dropdown",
                                        value="System Health",
                                    ),
                                    style={"textAlign": "center"},
                                ),
                                dcc.Graph(id="run-graph"),
                                html.H4("Per Simulated Run Data"),
                                dbc.Label("Select a run:"),
                                dbc.Input(
                                    id="run-dropdown",
                                    type="number",
                                    min=1,
                                    max=len(output.data),
                                    step=1,
                                    value=0,
                                ),
                                html.Br(),
                                html.Div(id="run-data-container", **SCROLL_DIV_KWARGS),
                                html.Br(),
                                html.H4("Per Run Prices"),
                                dbc.Checkbox(
                                    label="Show All", value=True, id="price-checkbox"
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
                                html.H2("Summary", style={"textAlign": "center"}),
                                html.P(
                                    "Summarized metrics and insights.",
                                    style={"textAlign": "center"},
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
                                html.H2("Configuration", style={"textAlign": "center"}),
                                html.P(
                                    "Simulation configuration.",
                                    style={"textAlign": "center"},
                                ),
                                html.H4("Price Parameters"),
                                html.P(
                                    f"Generative parameters for each stochastic process. Trained from {datetime.fromtimestamp(price_config['start'])} to {datetime.fromtimestamp(price_config['end'])}"
                                ),
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
                                    **SCROLL_DIV_KWARGS_50,
                                ),
                                html.Br(),
                                html.H5("Asset covariances"),
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
                                html.H4("External Liquidity Curves"),
                                html.P(
                                    f"Quotes from {metadata['template'].quotes_start} to {metadata['template'].quotes_end}."
                                ),
                                dbc.Button(
                                    "Fetch and show liquidity curve.",
                                    id="fetch-liquidity-button",
                                    className="mr-1",
                                ),  # TODO
                                html.Br(),
                                html.Br(),
                                html.H4("Module Metadata"),
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


@app.callback(
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
