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
            output = simulate(scenario, markets, num_iter=num_iter, ncpu=ncpu)[0]
        except Exception:  # pylint: disable=broad-except
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
    output = simulate(scenario, markets, num_iter=num_iter, ncpu=ncpu)[0]
    end = datetime.now()
    diff = end - start
    logger.info("Total runtime: %s", diff)
    return output


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

    _scenario = args.scenario
    _num_iter = args.num_iter

    if args.multiprocess:
        _ncpu = cpu_count()
    else:
        _ncpu = 1

    if args.analysis:
        _output = with_analysis(_scenario, MODELLED_MARKETS, _num_iter, _ncpu)
    else:
        _output = without_analysis(_scenario, MODELLED_MARKETS, _num_iter, _ncpu)

    logger.info("Done.")

    dir_ = os.path.join(RESULTS_DIR, _scenario)
    os.makedirs(dir_, exist_ok=True)
    i = len(os.listdir(dir_)) + 1
    fn = os.path.join(dir_, f"results_{_num_iter}_iters_{i}.pkl")
    with open(fn, "wb") as f:
        pickle.dump(_output, f)
