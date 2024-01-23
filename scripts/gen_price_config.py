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
import os
import argparse
from src.prices.utils import gen_price_config
from src.configs import ADDRESSES, get_scenario_config
from src.logging import get_logger


BASE_DIR = os.getcwd()
CONFIG_DIR = os.path.join(BASE_DIR, "src", "configs", "prices")
PLOTS_DIR = os.path.join(BASE_DIR, "figs", "prices")
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate price config for sim.")
    parser.add_argument("scenario", type=str, help="Scenario to simulate")
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot sample sim?", required=False
    )
    args = parser.parse_args()
    plot = args.plot

    scenario_cfg = get_scenario_config(args.scenario)

    freq = scenario_cfg["freq"]
    start = scenario_cfg["prices"]["start"]
    end = scenario_cfg["prices"]["end"]

    config_dir_w_freq = os.path.join(CONFIG_DIR, freq)
    plot_dir_w_freq = os.path.join(PLOTS_DIR, freq)

    os.makedirs(config_dir_w_freq, exist_ok=True)
    os.makedirs(plot_dir_w_freq, exist_ok=True)

    fn = os.path.join(config_dir_w_freq, f"{start}_{end}.json")
    plot_fn = os.path.join(plot_dir_w_freq, f"{start}_{end}.png")

    logger.info(
        "Generating price config for %s scenario with freq %s", args.scenario, freq
    )

    gen_price_config(fn, ADDRESSES, start, end, freq, plot=plot, plot_fn=plot_fn)
