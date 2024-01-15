"""
Generates the liquidity config file used to resample
crvUSD liquidity in StableSwap pools.
"""
import os
import argparse
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.configs import (
    STABLESWAP_ADDRESSES,
    MODELLED_MARKETS,
    LLAMMA_ALIASES,
    STABLESWAP_ALIASES,
)
from src.logging import get_logger
from src.plotting.sim import plot_debt_to_liquidity
from src.utils import (
    get_historical_user_snapshots,
    get_historical_stableswap_stats,
    group_user_states,
)


BASE_DIR = os.getcwd()
CONFIG_DIR = os.path.join(BASE_DIR, "src", "configs", "liquidity")
os.makedirs(CONFIG_DIR, exist_ok=True)


logger = get_logger(__name__)


def main(start: datetime, end: datetime, plot: bool) -> None:
    """
    Get the historical stableswap balances and user states in
    the given period.

    Compute the daily debt across all modelled markets, and the
    total crvUSD liquidity in the modelled stableswap pools.

    Determine the average ratio between crvUSD debt and crvUSD
    liquidity.

    Compute the generative parameters for the multivariate normal
    distribution of liquidity in each stableswap pool.

    Save the generative parameters and debt:liquidity ratio to a file.
    """
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    logger.info(
        "Generating liquidity config from %s to %s.",
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )

    stableswap_stats = get_historical_stableswap_stats(
        STABLESWAP_ADDRESSES, start_ts, end_ts
    )

    grouped_user_states = []
    for alias in MODELLED_MARKETS:
        market = LLAMMA_ALIASES[alias]
        user_states = get_historical_user_snapshots(market, start_ts, end_ts)
        grouped_user_states.append(group_user_states(user_states))

    debts = pd.DataFrame(
        [
            df.rename(columns={"debt": alias})[alias]
            for alias, df in zip(MODELLED_MARKETS, grouped_user_states)
        ]
    ).T
    debts["debt"] = debts.sum(axis=1)
    debts = debts.resample("1d").mean()

    liquidity = pd.DataFrame(
        [
            df.rename(columns={"crvUSD": STABLESWAP_ALIASES[address]})[
                STABLESWAP_ALIASES[address]
            ]
            for address, df in stableswap_stats.items()
        ]
    ).T
    liquidity["liquidity"] = liquidity.sum(axis=1)
    liquidity = liquidity.resample("1d").mean()

    ratio = (debts["debt"] / liquidity["liquidity"]).dropna()
    target_ratio = ratio.mean()
    stressed_ratio = ratio.quantile(0.99)

    config = {}
    for address, df in stableswap_stats.items():
        data = df[["peg", "crvUSD"]].values
        config[address] = {
            "mean_vector": np.mean(data, axis=0).tolist(),
            "covariance_matrix": np.cov(data, rowvar=False).tolist(),
        }

    config["target_ratio"] = target_ratio
    config["stressed_ratio"] = stressed_ratio

    fn = os.path.join(CONFIG_DIR, f"{start_ts}_{end_ts}.json")
    with open(fn, "w") as f:
        json.dump(config, f, indent=4)

    if plot:
        plot_debt_to_liquidity(debts, liquidity)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate liquidity config for sim.")
    parser.add_argument("start", type=str, help="ISO8601 start date.")
    parser.add_argument("end", type=str, help="ISO8601 end date.")
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot Debt:Liquidity", required=False
    )
    args = parser.parse_args()
    plot = args.plot
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)

    main(start, end, plot)
