"""
Provides a plotly Dash app to visualize simulation results.

Optionally run the simulation before creating the app.
"""

from datetime import datetime
import pickle
import json
import base64
from multiprocessing import cpu_count
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, dash_table, no_update
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from src.sim import run_scenario
from src.sim.results import MonteCarloResults
from src.logging import get_logger
from src.plotting.utils import make_square
from src.configs import ADDRESS_TO_SYMBOL
from .utils import load_markdown_file, clean_metadata

logger = get_logger(__name__)

TABLE_KWARGS = {
    "page_size": 10,
    "style_header": {"fontWeight": "bold"},
    "style_table": {"overflowX": "scroll", "tableLayout": "fixed"},
}

DIV_KWARGS = {
    "style": {"padding": "1%", "display": "inline-block", "width": "100%"},
}

TAB_KWARGS = {"label_style": {"color": "black"}}

global output
output = None

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

load_figure_template("flatly")

initial_modal = dbc.Modal(
    [
        dbc.ModalHeader("crvUSD Risk Simulator"),
        dbc.ModalBody(
            [
                html.P(
                    "crvUSD Risk is an Agent Based Model (ABM) that tests the resiliency of the crvUSD system under various market conditions."
                ),
                html.P("Please choose to run a simulation or load previous results."),
            ]
        ),
        dbc.ModalFooter(
            [
                dbc.Button(
                    "Run Simulation",
                    id="run-sim-button",
                    className="mr-1",
                    disabled=True,
                ),
                dcc.Upload(
                    id="upload-results",
                    children=dbc.Button(
                        "Load Results", id="load-results-button", className="mr-1"
                    ),
                ),
            ],
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


@app.callback(
    [Output("initial-modal", "is_open"), Output("main-content", "children")],
    [Input("upload-results", "contents")],
)
def load_results(contents):
    if contents is not None:
        global output
        output = pickle.loads(base64.b64decode(contents.split(",")[1]))
        content = _generate_content(output)
        return False, content
    return no_update


def _generate_content(output: MonteCarloResults):
    metadata = clean_metadata(output.metadata)
    price_config = metadata["template"].pricepaths.config

    layout = html.Div(
        [
            html.H1(
                "crvUSD Risk Simulation Results",
                style={"textAlign": "center", "padding-top": "2%"},
            ),
            html.Div(
                dbc.Alert(
                    [
                        html.H5("Overview"),
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
                                            "Markets: ", style={"font-weight": "bold"}
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
                                dcc.Dropdown(
                                    output.summary.columns,
                                    "arbitrageur_profit_max",
                                    id="aggregate-metric-dropdown",
                                ),
                                dcc.Graph(id="aggregate-graph"),
                                html.H4("Aggregate Data"),
                                dash_table.DataTable(
                                    data=output.summary.to_dict("records"),
                                    **TABLE_KWARGS,
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
                                html.H4("Per Simulated Run Plots"),
                                dcc.Dropdown(
                                    output.data[0].df.columns,
                                    "arbitrageur_profit",
                                    id="run-metric-dropdown",
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
                                dash_table.DataTable(id="run-data", **TABLE_KWARGS),
                                html.H4("Per Run Prices"),
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
                                html.H5(
                                    f"Generative parameters for each stochastic process. Trained from {datetime.fromtimestamp(price_config['start'])} to {datetime.fromtimestamp(price_config['end'])}"
                                ),
                                dash_table.DataTable(
                                    data=pd.DataFrame.from_dict(
                                        price_config["params"],
                                        orient="index",
                                    ).to_dict(orient="records"),
                                    **TABLE_KWARGS,
                                ),
                                html.Br(),
                                html.H5("Asset covariances"),
                                dash_table.DataTable(
                                    data=pd.DataFrame.from_dict(
                                        price_config["cov"],
                                        orient="index",
                                    )
                                    .reset_index(names="")
                                    .to_dict(orient="records"),
                                    **TABLE_KWARGS,
                                ),
                                html.Br(),
                                html.H5("Start Prices"),
                                dash_table.DataTable(
                                    data=pd.DataFrame.from_dict(
                                        price_config["curr_prices"],
                                        orient="index",
                                        columns=["Start Price"],
                                    ).to_dict(orient="records"),
                                    **TABLE_KWARGS,
                                ),
                                html.H4("External Liquidity Curves"),
                                html.P(
                                    f"Quotes from {metadata['template'].quotes_start} to {metadata['template'].quotes_end}."
                                ),
                                dbc.Button(
                                    "Fetch and show liquidity curve.",
                                    id="fetch-liquidity-button",
                                    className="mr-1",
                                ),  # TODO
                                html.H4("Module Metadata"),
                                dcc.Markdown(
                                    f"```json\n{json.dumps(metadata['template'].llamma.metadata, indent=4)}\n```"
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


@callback(Output("run-data", "data"), Input("run-dropdown", "value"))
def update_run_data_table(value: int):
    if not output:
        return no_update
    return output.data[value - 1].df.to_dict(orient="records")


@callback(Output("run-prices", "figure"), Input("run-dropdown", "value"))
def update_run_prices(value: int):
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


def sim(num_iter: int) -> MonteCarloResults:
    start = datetime.now()
    output = run_scenario("baseline", "wstETH", num_iter=num_iter, ncpu=cpu_count())
    end = datetime.now()
    diff = end - start
    logger.info("Done. Total runtime: %s", diff)
    return output


if __name__ == "__main__":
    ### UNCOMMENT TO RUN LOCAL SIM ###
    # output: MonteCarloResults = sim(num_iter=10)
    # with open(f"{output.metadata.scenario}.pkl", "wb") as f:
    #     pickle.dump(output, f)

    app.run()
