"""
Provides a plotly Dash app to visualize simulation results.
"""
from datetime import datetime
import pickle
import json
import os
import pandas as pd
import numpy as np
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
)
from dash import dash_table
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from src.sim.results import MonteCarloResults
from src.plotting.utils import make_square
from src.configs import ADDRESS_TO_SYMBOL, LLAMMA_ALIASES, TOKEN_DTOs
from src.utils import get_quotes
from src.metrics.utils import entity_str
from app.utils import (
    load_markdown_file,
    clean_metadata,
    load_results,
    plot_quotes,
    plot_regression,
    create_card,
    list_experiments,
    list_param_sweeps,
    list_scenarios,
)

PORT = os.getenv("PORT", "8050")

plt.switch_backend("Agg")

DECIMALS = 5  # number of decimal places to round to

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

QUANTILE = 0.99

global output
output = None

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

load_figure_template("flatly")

initial_modal = dbc.Modal(
    [
        dbc.ModalHeader(html.H2("crvUSD Risk Dashboard")),
        dbc.ModalBody(
            [
                html.P(
                    "crvUSD Risk is an Agent Based Model (ABM) "
                    "that tests the resiliency of the crvUSD system "
                    "under various market conditions."
                ),
                html.P(
                    "Please choose a scenario to view results for. "
                    "Experiments define the parameter sets being tested."
                    "The generic experiment uses the default parameters."
                ),
                dbc.Label(
                    "Select experiment",
                    html_for="select-experiment",
                ),
                dbc.Select(
                    id="select-experiment",
                    options=[
                        {"label": exp.title(), "value": exp}
                        for exp in list_experiments()
                    ],
                    value="generic",
                ),
                html.Br(),
                dbc.Label(
                    "Select parameters",
                    html_for="select-parameter",
                ),
                dbc.Select(id="select-parameter", disabled=True),
                html.Br(),
                dbc.Label(
                    "Select scenario",
                    html_for="select-scenario",
                ),
                dbc.Select(id="select-scenario", disabled=True),
                html.Br(),
                dbc.Button(
                    "Load Results",
                    id="load-results-button",
                    className="mr-1",
                    disabled=True,
                    n_clicks=None,
                ),
                html.Hr(),
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
        html.Div(id="main-content", style={"padding-top": "1%"}),
    ]
)


@callback(
    [
        Output("select-parameter", "options"),
        Output("select-parameter", "disabled"),
    ],
    Input("select-experiment", "value"),
    State("initial-modal", "is_open"),
)
def update_parameter_dropdown(experiment, is_open):
    if not is_open or experiment is None:
        return no_update, no_update

    params = list_param_sweeps(experiment)
    # Do some label formatting
    if experiment == "debt_ceilings":
        labels = [p + "x" for p in params]
    elif experiment in ["fees", "chainlink_limits"]:
        labels = [f"{p.replace('_', '.')}%" for p in params]
    elif experiment == "generic":
        labels = ["Default Parameters"]
    else:
        labels = params

    options = [{"label": label, "value": param} for label, param in zip(labels, params)]

    return options, False


@callback(
    [
        Output("select-scenario", "options"),
        Output("select-scenario", "disabled"),
        Output("load-results-button", "disabled"),
    ],
    [
        Input("select-experiment", "value"),
        Input("select-parameter", "value"),
    ],
    State("initial-modal", "is_open"),
)
def update_scenario_dropdown(experiment, param, is_open):
    if not is_open or param is None or experiment is None:
        return no_update, no_update, no_update

    params = list_scenarios(experiment, param)
    labels = [p.replace(".pkl", "").replace("_", " ").title() for p in params]

    options = [{"label": label, "value": param} for label, param in zip(labels, params)]

    return options, False, False


@callback(
    [
        Output("initial-modal", "is_open"),
        Output("main-content", "children"),
        Output("loading-simulation", "children"),
    ],
    [
        Input("load-results-button", "n_clicks"),
    ],
    [
        State("initial-modal", "is_open"),
        State("select-experiment", "value"),
        State("select-parameter", "value"),
        State("select-scenario", "value"),
    ],
)
def generate_content(
    nclicks,
    is_open,
    experiment,
    parameter,
    scenario,
):
    """
    Generate main html content from the
    loaded results.
    """
    if nclicks is None:
        return is_open, no_update, no_update
    else:
        global output
        output = load_results(experiment, parameter, scenario)

    return False, _generate_content(output), no_update


def _generate_content(output: MonteCarloResults):
    output.summary  # force compute and cache

    metadata = clean_metadata(output.metadata)
    template = metadata["template"]
    price_config = template.pricepaths.config.copy()
    for k, v in price_config["params"].items():
        v.update({"start_price": price_config["curr_prices"][k]})

    market_options = [
        {"label": alias, "value": LLAMMA_ALIASES[alias]}
        for alias in template.market_names
    ]

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
        for asset in [c.address for c in template.coins]
    ]

    worst_depeg_agg = output.summary.loc[
        output.summary[["Aggregator Price Min", "Aggregator Price Max"]]
        .subtract(1)
        .abs()
        .stack()
        .idxmax()
    ]

    cov_matrix = pd.DataFrame(price_config["cov"])
    std_devs = np.sqrt(np.diag(cov_matrix))
    std_dev_matrix = np.outer(std_devs, std_devs)
    corr_matrix = cov_matrix / std_dev_matrix
    corr_matrix = pd.DataFrame(
        corr_matrix, index=cov_matrix.index, columns=cov_matrix.columns
    )

    var_body = html.Div(
        [
            html.H4(
                f"VaR: {output.summary['Bad Debt Pct Max'].quantile(QUANTILE):,.2f} %"
            )
        ]
    )

    lar_body = html.Div(
        [
            html.H4(
                f"LaR: {output.summary['Debt Liquidated Pct Max'].quantile(QUANTILE):,.2f} %"
            )
        ]
    )

    for controller in template.controllers:
        _controller = entity_str(controller, "controller")

        controller_bad_debt = output.summary[
            f"Bad Debt Pct On {_controller} Max"
        ].quantile(QUANTILE)
        var_body.children.append(
            html.H6(f"VaR on {_controller}: {controller_bad_debt:,.2f} %")
        )

        collateral_liquidated = output.summary[
            f"Debt Liquidated Pct On {entity_str(controller, 'controller')} Max"
        ].quantile(QUANTILE)
        lar_body.children.append(
            html.H6(f"LaR on {_controller}: {collateral_liquidated:,.2f} %")
        )

    var_card = create_card(
        "Value at Risk",
        "Value at Risk (VaR) is the p99 maximum bad debt observed over the simulated runs as a percentage of simulated debt. This may intuitively be interpreted as: Bad d under the input assumptions will only ever exceed VaR 1% of the time.",
        var_body,
    )

    lar_card = create_card(
        "Liquidations at Risk",
        "Liquidations at Risk (LaR) is the p99 maximum debt liquidated over the simulated runs as a percentage of simulated debt. This may intuitively be interpreted as: Liquidated debt under the input assumptions will only ever exceed LaR 1% of the time.",
        lar_body,
    )

    blar_body = html.Div(
        [
            html.H4(
                f"BLaR: {output.summary['Borrower Loss Pct Max'].quantile(QUANTILE):,.2f} %"
            )
        ]
    )
    blar_card = create_card(
        "Borrower Losses at Risk",
        "Borrower Losses at Risk (BLaR) is the p99 maximum borrower losses observed over the simulated runs as a percentage of simulated debt. This includes both LVR and net liquidation losses (i.e. collateral - debt liquidated). This may intuitively be interpreted as: Borrower losses under the input assumptions will only ever exceed BLaR 1% of the time.",
        blar_body,
    )

    spools = [
        entity_str(spool, "stableswap").title() for spool in template.stableswap_pools
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
            html.H4(f"Worst Depeg in Aggregator: {worst_depeg_agg:,.3f}"),
            html.H4(
                f"Worst Depeg in StableSwap: {worst_depeg_spool_val:,.3f} in {worst_depeg_spool}"
            ),
        ]
    )
    depeg_card = create_card(
        "Worst Depeg",
        "The worst depeg (up or down) observed over the simulated runs for the Aggregator or any StableSwap pool.",
        depeg_str,
    )

    metric_cards = dbc.CardGroup([var_card, lar_card, blar_card, depeg_card])

    overview_cards = dbc.CardGroup(
        [
            create_card(
                "Scenario Name",
                metadata["scenario"].title(),
                "",
                color="primary",
                border=False,
            ),
            create_card(
                "Number of Iterations",
                len(output.data),
                "",
                color="primary",
                border=False,
            ),
            create_card(
                "Simulation Horizon",
                f"{metadata['num_steps']} steps of {metadata['freq']}",
                "",
                color="primary",
                border=False,
            ),
            create_card(
                "Markets", str(metadata["markets"]), "", color="primary", border=False
            ),
        ]
    )

    layout = html.Div(
        [
            html.H1(
                "crvUSD Risk Dashboard",
                style={"textAlign": "center"},
            ),
            html.P(
                "Not financial advice. All assumptions and limitations documented in the INFO tab.",
                style={"textAlign": "center"},
            ),
            html.Div(
                dbc.Alert(
                    overview_cards,
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
                                metric_cards,
                                html.Br(),
                                html.H4(
                                    "Metric Histograms", style={"textAlign": "center"}
                                ),
                                dbc.Row(
                                    dbc.Col(
                                        dbc.Select(
                                            options=aggregate_columns,
                                            id="aggregate-metric-dropdown",
                                            value="System Health Min",
                                        ),
                                        width=4,
                                    ),
                                    justify="center",
                                ),
                                dbc.Spinner(
                                    dcc.Graph(id="aggregate-graph"),
                                ),
                                html.H4(
                                    "Aggregate Data", style={"textAlign": "center"}
                                ),
                                html.Br(),
                                html.Div(
                                    dash_table.DataTable(
                                        columns=[
                                            {"name": i, "id": i}
                                            for i in output.summary.columns
                                        ],
                                        data=output.summary.reset_index(
                                            names=["Run ID"]
                                        )
                                        .round(DECIMALS)
                                        .to_dict("records"),
                                        # sort_action="native",
                                        page_action="native",
                                        page_size=10,
                                        id="aggregate-data",
                                        style_table={"overflowX": "auto"},
                                        style_cell={
                                            "padding-left": "10px",
                                            "padding-right": "10px",
                                            "padding-top": "20px",
                                            "padding-bottom": "20px",
                                            "color": "white",
                                        },
                                        style_header={
                                            "fontWeight": "bold",
                                            "color": "white",
                                            "backgroundColor": "#2c3d4f",
                                        },
                                        style_data_conditional=[
                                            {
                                                "if": {"row_index": "odd"},
                                                "backgroundColor": "#364858",
                                            },
                                            {
                                                "if": {"row_index": "even"},
                                                "backgroundColor": "#2c3d4f",
                                            },
                                        ],
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
                                            value="Debt Liquidated Pct",
                                        ),
                                        width=4,
                                    ),
                                    justify="center",
                                ),
                                dbc.Spinner(
                                    dcc.Graph(id="run-graph"),
                                ),
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
                                                min=0,
                                                max=len(output.data) - 1,
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
                                            value=False,
                                            id="price-checkbox",
                                        ),
                                        width=1,
                                    ),
                                    justify="center",
                                ),
                                dbc.Spinner(
                                    dcc.Graph(id="run-prices"),
                                ),
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
                                    "Asset Correlations", style={"textAlign": "center"}
                                ),
                                html.Div(
                                    dbc.Table.from_dataframe(
                                        corr_matrix.reset_index(names="Name").round(
                                            DECIMALS
                                        ),
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
                                        dbc.Spinner(
                                            html.Div(
                                                id="fetch-liquidity-output",
                                            ),
                                        ),
                                    ],
                                    style={"textAlign": "center"},
                                ),
                                html.Br(),
                                html.Br(),
                                html.H4(
                                    "Market Metadata", style={"textAlign": "center"}
                                ),
                                dbc.Select(
                                    options=market_options,
                                    id="market-metadata-dropdown",
                                    value=market_options[0]["value"],
                                ),
                                html.Div(id="market-metadata-container"),
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
                                    load_markdown_file("./app/assumptions.md"),
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
    fig = px.histogram(dff)
    fig.update_layout(bargap=0.1)
    return fig


N = 3  # keep every N rows for run prices


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
    if not output or value is None:
        return no_update
    return dbc.Table.from_dataframe(
        output.data[value].df.reset_index(names=["Time"]).round(DECIMALS),
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
    start = output.metadata["template"].quotes_start
    end = output.metadata["template"].quotes_end
    in_asset_dto = TOKEN_DTOs[in_asset]
    out_asset_dto = TOKEN_DTOs[out_asset]
    pair = tuple(sorted((in_asset_dto, out_asset_dto)))
    quotes = get_quotes(
        start,
        end,
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
                        dcc.Graph(figure=plot_quotes(df, in_asset_dto, out_asset_dto)),
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
    if not output or value is None:
        return no_update
    df = output.data[value].pricepaths.prices
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
                        prices = _df.pricepaths.prices.iloc[::N, :]
                        fig.add_trace(
                            go.Scatter(x=prices.index, y=prices[col], mode="lines"),
                            row=i + 1,
                            col=j + 1,
                        )
                else:
                    _df = df.iloc[::N, :]
                    fig.add_trace(
                        go.Scatter(x=_df.index, y=_df[col], mode="lines"),
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


@callback(
    Output("market-metadata-container", "children"),
    Input("market-metadata-dropdown", "value"),
)
def update_metadata_container(value):
    if output is None:
        return no_update

    llamma = None
    for _llamma in output.metadata["template"].llammas:
        if _llamma.address == value:
            llamma = _llamma
            break

    return (
        html.Div(
            dcc.Markdown(f"```json\n{json.dumps(llamma.metadata, indent=4)}\n```"),
            **SCROLL_DIV_KWARGS,
        ),
    )


if __name__ == "__main__":
    app.run(port=PORT, host="0.0.0.0")
