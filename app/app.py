"""
Provides a plotly Dash app to visualize simulation results.

Optionally run the simulation before creating the app.
"""

from datetime import datetime
import pickle
from multiprocessing import cpu_count
from dash import Dash, html, dcc, callback, Output, Input, dash_table
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


def create_app(output: MonteCarloResults):
    """Create a Dash app to visualize simulation results."""
    app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

    load_figure_template("flatly")

    metadata = output.metadata

    app.layout = html.Div(
        [
            html.H1(
                "crvUSD Risk Simulation Results",
                style={"textAlign": "center", "padding-top": "2%"},
            ),
            html.Div(
                dbc.Alert(
                    [
                        html.P("Exploring results for the below simulation run:"),
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
                                            "Markets: ", style={"font-weight": "bold"}
                                        ),
                                        str(metadata["markets"]),
                                    ]
                                ),
                            ]
                        ),
                        html.P("We provide two views of the data:"),
                        html.Ul(
                            [
                                html.Li(
                                    [
                                        html.Span(
                                            "Summary: ", style={"font-weight": "bold"}
                                        ),
                                        "Aggregated data across all runs. We provide histograms of key metrics and the raw data.",
                                    ]
                                ),
                                html.Li(
                                    [
                                        html.Span(
                                            "Per Simulated Run: ",
                                            style={"font-weight": "bold"},
                                        ),
                                        "Data for each individual run. We provide plots of key metrics over all runs, the raw data, and the prices used for each run.",
                                    ]
                                ),
                            ]
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
                                html.H4("Summary Metric Histograms"),
                                dcc.Dropdown(
                                    output.summary.columns,
                                    "arbitrageur_profit_max",
                                    id="summary-metric-dropdown",
                                ),
                                dcc.Graph(id="summary-graph"),
                                html.H4("Summary Data"),
                                dash_table.DataTable(
                                    data=output.summary.to_dict("records"),
                                    **TABLE_KWARGS,
                                ),
                            ],
                            **DIV_KWARGS,
                        ),
                        label="Summary Data",
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
                ],
            ),
        ]
    )

    @callback(
        Output("summary-graph", "figure"), Input("summary-metric-dropdown", "value")
    )
    def update_summary_graph(value):
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
        return output.data[value - 1].df.to_dict(orient="records")

    @callback(Output("run-prices", "figure"), Input("run-dropdown", "value"))
    def update_run_prices(value: int):
        df = output.data[value - 1].pricepaths.prices
        cols = [col for col in df.columns if col != "timestamp"]
        titles = [ADDRESS_TO_SYMBOL[col] for col in cols]
        n, m = make_square(len(cols))
        fig = make_subplots(
            rows=n, cols=m, start_cell="bottom-left", subplot_titles=titles
        )

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

    app.run()

    return app


def sim(num_iter: int = 10) -> MonteCarloResults:
    start = datetime.now()
    output = run_scenario("baseline", "wstETH", num_iter=num_iter, ncpu=cpu_count())
    end = datetime.now()
    diff = end - start
    logger.info("Done. Total runtime: %s", diff)
    return output


if __name__ == "__main__":
    ### UNCOMMENT TO RUN LOCAL SIM ###
    # output: MonteCarloResults = sim(num_iter=1000)
    # with open("sample_output.pkl", "wb") as f:
    #     pickle.dump(output, f)

    ### UNCOMMENT TO READ LOCAL RESULTS ###
    with open("sample_output.pkl", "rb") as f:
        output: MonteCarloResults = pickle.load(f)

    create_app(output)
