"""
Script to run a Monte Carlo simulation.
"""
from typing import List
import os
import cProfile
import pstats
import pdb
import argparse
import pickle
from datetime import datetime
from multiprocessing import cpu_count
from src.sim import simulate
from src.logging import get_logger
from src.sim.results import MonteCarloResults
from src.configs import MODELLED_MARKETS

logger = get_logger(__name__)

BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def with_analysis(
    scenario: str, markets: List[str], num_iter: int, ncpu: int
) -> MonteCarloResults:
    """
    Run simulation with profiling and debugging.
    """
    with cProfile.Profile() as pr:
        try:
            global output
            output = simulate(scenario, markets, num_iter=num_iter, ncpu=ncpu)[0]
        except Exception:
            pdb.post_mortem()
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats()
    stats.dump_stats("./logs/sim.prof")
    return output


def without_analysis(
    scenario: str, markets: List[str], num_iter: int, ncpu: int
) -> MonteCarloResults:
    """
    Run simulation without any analysis.
    """
    start = datetime.now()
    global output
    output = simulate(scenario, markets, num_iter=num_iter, ncpu=ncpu)[0]
    end = datetime.now()
    diff = end - start
    logger.info("Total runtime: %s", diff)
    return output


def analysis_help() -> None:
    """
    Help for analysis.
    """
    print("Call `output.summary` for a DF of summary metrics.")
    print("Call `output.plot_runs(<metric_id>)` to plot the input metric for all runs.")
    print("Call `output.metric_map` to get a list of metrics and their ids.")
    print("Call `output.plot_summary()` to plot histograms of summary metrics.")
    print("Call `output.data[i].plot_prices()` to plot the prices used for run `i`.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate price config for sim.")
    parser.add_argument(
        "scenario", type=str, help="Scenario to simulate (from src/configs/scenarios)."
    )
    parser.add_argument("num_iter", type=int, help="Number of runs.", default=10)
    parser.add_argument(
        "-mp",
        "--multiprocess",
        action="store_true",
        help="Multiprocess?",
        required=False,
    )
    parser.add_argument(
        "-a",
        "--analysis",
        action="store_true",
        help="Analyze Runtime?",
        required=False,
    )
    args = parser.parse_args()

    scenario = args.scenario
    num_iter = args.num_iter

    if args.multiprocess:
        ncpu = cpu_count()
    else:
        ncpu = 1

    if args.analysis:
        output = with_analysis(scenario, MODELLED_MARKETS, num_iter, ncpu)
    else:
        output = without_analysis(scenario, MODELLED_MARKETS, num_iter, ncpu)

    logger.info("Done. Call `analysis_help()` for more info in interactive mode.")

    dir_ = os.path.join(RESULTS_DIR, scenario)
    os.makedirs(dir_, exist_ok=True)
    i = len(os.listdir(dir_)) + 1
    fn = os.path.join(dir_, f"results_{num_iter}_iters_{i}.pkl")
    with open(fn, "wb") as f:
        pickle.dump(output, f)
