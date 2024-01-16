"""
Plot the simulated price impact curve for each [directional]
permutation of modeled tokens. These are based off the 
1inch quotes we store in our postgres database.
"""
import pandas as pd
from src.logging import get_logger
from src.modules import ExternalMarket
from src.sim.scenario import Scenario
from src.plotting import plot_regression, plot_predictions

PATH = "figs/price_impacts"
FN_REGR = f"{PATH}/regressions/" + "{}_{}.png"
FN_PRED = f"{PATH}/predictions/" + "{}_{}.png"

logger = get_logger(__name__)


def plot(
    quotes: pd.DataFrame,
    market: ExternalMarket,
    i: int,
    j: int,
    scale: str = "log",
    fn_regr: str | None = None,
    fn_pred: str | None = None,
) -> None:
    """Generate relevant price impact plots"""
    in_token = market.coins[i]
    out_token = market.coins[j]

    quotes_ = quotes.loc[in_token.address, out_token.address]
    _ = plot_regression(quotes_, i, j, market, scale=scale, fn=fn_regr)
    _ = plot_predictions(quotes_, i, j, market, scale=scale, fn=fn_pred)


def main() -> None:
    """Generate price impact plots for each token pair."""
    scenario = Scenario("baseline", ["wsteth"])
    scenario.update_market_prices(scenario.pricepaths[0])

    quotes = scenario.generate_markets()  # regenerate to fetch quotes

    # This takes a while to plot
    i = 1
    n = len(scenario.pairs) * 2
    for pair in scenario.pairs:
        market = scenario.markets[pair]
        token1, token2 = pair

        logger.info(f"Plotting ({token1.symbol, token2.symbol}). ({i}/{n})")
        plot(
            quotes,
            market,
            0,
            1,
            scale="log",
            fn_regr=FN_REGR.format(token1.symbol, token2.symbol),
            fn_pred=FN_PRED.format(token1.symbol, token2.symbol),
        )

        i += 1

        logger.info(f"Plotting ({token2.symbol, token1.symbol}). ({i}/{n})")
        plot(
            quotes,
            market,
            1,
            0,
            scale="log",
            fn_regr=FN_REGR.format(token2.symbol, token1.symbol),
            fn_pred=FN_PRED.format(token2.symbol, token1.symbol),
        )

        i += 1


if __name__ == "__main__":
    main()
