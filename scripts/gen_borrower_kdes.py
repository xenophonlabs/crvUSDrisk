"""
Generate the Kernel Density Estimation (KDE) instances
for each crvUSD market. The KDEs are trained on user
state snapshots from the crvUSD subgraph.
"""
import os
import argparse
import pickle
from datetime import datetime
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from src.configs import LLAMMA_ALIASES, ALIASES_LLAMMA
from src.logging import get_logger
from src.plotting.sim import plot_borrowers_2d
from src.utils import get_historical_user_snapshots


BASE_DIR = os.getcwd()
CONFIG_DIR = os.path.join(BASE_DIR, "src", "configs", "borrowers")
os.makedirs(CONFIG_DIR, exist_ok=True)


logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate borrower kde for sim.")
    parser.add_argument("market", type=str, help="Market to generate kde for.")
    parser.add_argument("start", type=str, help="ISO8601 start date.")
    parser.add_argument("end", type=str, help="ISO8601 end date.")
    parser.add_argument(
        "-p", "--plot", action="store_true", help="Plot KDE", required=False
    )
    args = parser.parse_args()
    plot = args.plot
    market = args.market
    start = datetime.fromisoformat(args.start)
    end = datetime.fromisoformat(args.end)
    start_ts = int(start.timestamp())
    end_ts = int(end.timestamp())

    if "0x" in market:
        address = market
        alias = ALIASES_LLAMMA[market]
    else:
        address = LLAMMA_ALIASES[market]
        alias = market

    logger.info(
        "Generating KDE for market %s from %s to %s.",
        alias,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )

    df = get_historical_user_snapshots(address, start_ts, end_ts)

    dir = os.path.join(CONFIG_DIR, alias.lower())

    os.makedirs(dir, exist_ok=True)
    fn = os.path.join(dir, f"{start_ts}_{end_ts}.pkl")

    kde = gaussian_kde(df[["debt_log", "collateral_log", "n"]].values.T)
    with open(fn, "wb") as f:
        pickle.dump(kde, f)

    if plot:
        values = df[["health", "collateral_log"]].values.T
        density = gaussian_kde(values)(values)
        health, collateral_log = values
        title = f"Borrower States for {alias} from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
        fig = plot_borrowers_2d(health, collateral_log, density, title=title)
        plt.show()
