"""
Script to run a parameter sweep.
"""
from typing import List, Dict, Any
import os
import pickle
from datetime import datetime
from multiprocessing import cpu_count
from src.sim import simulate
from src.logging import get_logger
from src.sim.results import MonteCarloResults
from src.configs import MODELLED_MARKETS
from src.configs.parameters import DEBT_CEILING_SWEEP

logger = get_logger(__name__)

BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run(
    scenario: str,
    markets: List[str],
    num_iter: int,
    ncpu: int,
    to_sweep: List[Dict[str, Any]],
) -> List[MonteCarloResults]:
    """
    Run simulation without any analysis.
    """
    start = datetime.now()
    outputs = simulate(
        scenario, markets, num_iter=num_iter, ncpu=ncpu, to_sweep=to_sweep
    )
    end = datetime.now()
    diff = end - start
    logger.info("Total runtime: %s", diff)
    return outputs


def save(
    scenario: str,
    swept_var: str,
    swept_val: Any,
    output: MonteCarloResults,
) -> None:
    """
    Save scenario results.
    """
    dir_ = os.path.join(RESULTS_DIR, f"sweep_{swept_var}_{swept_val}", scenario)
    os.makedirs(dir_, exist_ok=True)
    i = len(os.listdir(dir_)) + 1
    fn = os.path.join(dir_, f"results_{i}.pkl")
    with open(fn, "wb") as f:
        pickle.dump(output, f)


scenarios = [
    # "baseline"
    # "adverse vol"
    "severe vol"
    # # "adverse drift"
    # # "severe drift"
    # "adverse growth"
    # "severe growth"
    # "adverse crvusd liquidity"
    # "severe crvusd liquidity"
    # "adverse flash crash"
    # "severe flash crash"
    # "adverse depeg"
    # "severe depeg"
    # # "severe vol and adverse drift"
    # # "severe vol and severe drift"
    "severe vol and adverse growth"
    "severe vol and severe growth"
    # "severe vol and adverse crvusd liquidity"
    # "severe vol and severe crvusd liquidity"
    "adverse flash crash and adverse growth"
    "adverse flash crash and severe growth"
    # "adverse flash crash and adverse crvusd liquidity"
    # "adverse flash crash and severe crvusd liquidity"
]

to_sweep = DEBT_CEILING_SWEEP

if __name__ == "__main__":
    num_iter = 100
    num_rounds = 3
    ncpu = cpu_count()

    for _ in range(num_rounds):
        for scenario in scenarios:
            try:
                outputs = run(
                    scenario, MODELLED_MARKETS, num_iter, ncpu, to_sweep=to_sweep
                )
                for output, params in zip(outputs, to_sweep):
                    assert len(params) == 1, "Only handle sweeping one param at a time."
                    swept_var = list(params.keys())[0]
                    swept_val = params[swept_var]
                    save(scenario, swept_var, swept_val, output)
            except Exception as e:
                logger.critical("Failed scenario %s", scenario)

    logger.info("Done.")
