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

logger = get_logger(__name__)

BASE_DIR = os.getcwd()
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run(
    scenario: str,
    markets: List[str],
    num_iter: int,
    to_sweep: List[Dict[str, Any]],
    ncpu: int,
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
    experiment: str,
    scenario: str,
    output: MonteCarloResults,
    swept_var: str | None = None,
    swept_val: Any | None = None,
) -> None:
    """
    Save scenario results.
    """
    if swept_var:
        dir_ = os.path.join(RESULTS_DIR, experiment, f"sweep_{swept_var}_{swept_val}")
    else:
        dir_ = os.path.join(RESULTS_DIR, experiment, "no_sweep")

    os.makedirs(dir_, exist_ok=True)
    fn = os.path.join(dir_, f"{scenario.replace(' ', '_')}.pkl")
    with open(fn, "wb") as f:
        pickle.dump(output, f)


def sweep(
    experiment: str,
    scenarios: List[str],
    num_iter: int,
    to_sweep: List[Dict[str, Any]],
    ncpu: int = cpu_count(),
) -> None:
    for scenario in scenarios:
        try:
            outputs = run(scenario, MODELLED_MARKETS, num_iter, to_sweep, ncpu)
            for output, params in zip(outputs, to_sweep):
                if params:
                    assert len(params) == 1, "Only handle sweeping one param at a time."
                    swept_var = list(params.keys())[0]
                    swept_val = params[swept_var]
                    save(experiment, scenario, output, swept_var, swept_val)
                else:
                    save(experiment, scenario, output)
        except Exception as e:
            logger.critical("Failed scenario %s with exception %s", scenario, str(e))

    logger.info("Done.")
