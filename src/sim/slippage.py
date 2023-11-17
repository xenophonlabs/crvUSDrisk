from typing import List
import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.modules.market import ExternalMarket
from src.plotting import plot_predictions, plot_price_impact_prediction_error


def regress(df, x_vars, y_var, v=False):
    """
    Perform a simple OLS regression of the form:

        y = c dot x + b

    where c is a vector of coefficients, x is a
    vector of input variables, and b is the intercept.

    Parameters
    ----------
    df : pd.DataFrame
        Trades
    x_vars : List[str]
        Input variables
    y_var : str
        Output variable
    v : Optional[bool]
        Whether to print OLS results.

    Returns
    -------
    OLSResults
        The results object for the statsmodels OLS
        method. Contains params and rsquared value.
    """
    X = sm.add_constant(df[x_vars])
    y = df[y_var]
    model = sm.OLS(y, X).fit()
    if v:
        print(model.summary())
    return model


# === ARCHIVED === #
# We used to analyze uniswap trades here to examine slippage.
# We now use the output from our 1inch API calls.


def process_trades(df: pd.DataFrame, decimals: List[int]) -> List[pd.DataFrame]:
    """
    Deprecated.
    Process trade data for a given market.

    Parameters
    ----------
    df : pd.DataFrame
        Unprocessed trades
    decimals : List[int]
        Decimals of tokens in market.

    Returns
    -------
    pd.DataFrame
        Processed trades where token0 is being sold
    pd.DataFrame
        Processed trades where token1 is being sold
    """
    df = df.copy()

    price_cols = ["price_implied", "price_actual", "previous_price_actual"]

    # NOTE these adjustments are stupid, the data is coming in wrong.
    # The amounts should be adhering to the decimals of each token.
    # Instead, the amounts are just all fucked up.

    # Make sure price is token_out for token_in instead of
    # amount0 for amount1
    cond = df["amount1_adjusted"] < 0
    for col in price_cols:
        df.loc[cond, col] = df.loc[cond, col].apply(lambda x: x / 10 ** sum(decimals))
        df.loc[~cond, col] = df.loc[~cond, col].apply(
            lambda x: 1 / x * 10 ** sum(decimals)
        )

    df["amount0_adjusted"] *= 10 ** (decimals[0] - decimals[1])
    df["amount1_adjusted"] /= 10 ** (decimals[0] - decimals[1])

    df["price_impact"] = (df["price_implied"] - df["previous_price_actual"]) / (
        df["previous_price_actual"]
    )

    # Date cols
    df["date"] = pd.to_datetime(df["evt_block_time"]).dt.date
    df.sort_values(
        by="evt_block_time",
        axis=0,
        ascending=True,
        inplace=True,
        kind="quicksort",
        na_position="last",
    )

    return df[df["amount0_adjusted"] > 0], df[df["amount0_adjusted"] < 0]


def analyze(fn, decimals, plot=False, return_dfs=False):
    """Deprecated."""
    with open(fn, "r") as f:
        df = pd.read_csv(f)

    df0, df1 = process_trades(df, decimals)

    coefs = np.zeros((2, 2))
    intercepts = np.zeros((2, 2))
    results = np.zeros((2, 2))

    ols = regress(df0, x_vars=["amount0_adjusted"], y_var=["price_impact"], v=False)
    b, m = ols.params
    coefs[0][1] = m
    intercepts[0][1] = b
    results[0][1] = ols.rsquared_adj

    ols = regress(df1, x_vars=["amount1_adjusted"], y_var=["price_impact"], v=False)
    b, m = ols.params
    coefs[1][0] = m
    intercepts[1][0] = b
    results[1][0] = ols.rsquared_adj

    if plot:
        market = ExternalMarket(2, coefs, intercepts)

        df0["predicted"] = df0.apply(
            lambda row: -market.trade(
                row["amount0_adjusted"], row["previous_price_actual"], 0, 1
            ),
            axis=1,
        )
        df0["pct_error"] = df0.apply(
            lambda row: -abs(row["amount1_adjusted"] - row["predicted"])
            / row["amount1_adjusted"]
            * 100,
            axis=1,
        )
        print(f"Mean percentage error: {df0['pct_error'].mean():.3f}%")

        df1["predicted"] = df1.apply(
            lambda row: -market.trade(
                row["amount1_adjusted"], row["previous_price_actual"], 1, 0
            ),
            axis=1,
        )
        df1["pct_error"] = df1.apply(
            lambda row: -abs(row["amount0_adjusted"] - row["predicted"])
            / row["amount0_adjusted"]
            * 100,
            axis=1,
        )
        print(f"Mean percentage error: {df1['pct_error'].mean():.3f}%")

        plot_predictions(df0, df1)
        plot_price_impact_prediction_error(df0, "amount0_adjusted")
        plot_price_impact_prediction_error(df1, "amount1_adjusted")

    out = [coefs, intercepts, results]

    if return_dfs:
        out.extend([df0, df1])

    # TODO where do we want to save these results?
    return out
