"""
Generate the config file for simulating prices.
This config contains the stochastic process parameters
for generating prices for each coin, as well as the 
covariance between them.

Parameters for Stablecoins:
    `theta` = Rate of mean reversion
    `mu` = Long-term mean
    `sigma` = Volatility
    `Type` = Ornstein-Uhlenbeck

Parameters for Non-Stablecoins:
    `mu` = Drift
    `sigma` = Volatility
    `type` = GBM

We default to a 1h granularity in price data, sampled over
the last 60 days.
"""
import argparse
from datetime import datetime, timedelta
from src.prices.utils import gen_price_config
from src.configs import ADDRESSES
from src.logging import get_logger


logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate price config for sim.")
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot sample sim?", required=False
    )
    args = parser.parse_args()
    plot = args.plot

    freq = "1h"
    end = int(
        (datetime.now() - timedelta(hours=1)).timestamp()
    )  # Offset by an hour to prevent missing data
    start = int((datetime.now() - timedelta(days=60)).timestamp())
    fn = f"./src/configs/prices/{freq}_{start}_{end}.json"

    gen_price_config(fn, ADDRESSES, start, end, freq=freq, plot=plot)
