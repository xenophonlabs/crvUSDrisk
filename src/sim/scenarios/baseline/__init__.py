"""
Provides the function to simulate the 
baseline scenario.
"""

from typing import List, Type
from copy import deepcopy
import pickle
import multiprocessing as mp
from .strategy import BaselineStrategy
from ...processing import MonteCarloProcessor
from ...results import MonteCarloResults
from ...scenario import Scenario
from ....logging import (
    get_logger,
    multiprocessing_logging_queue,
    configure_multiprocess_logging,
)
from ....metrics import DEFAULT_METRICS, Metric
from ....configs import STABLE_CG_IDS


logger = get_logger(__name__)


# pylint: disable=too-many-arguments, too-many-locals
def simulate(
    config: str,
    market_name: str,
    num_iter: int = 1,
    metrics: List[Type[Metric]] | None = None,
    local: str = "",
    ncpu: int = mp.cpu_count(),
) -> MonteCarloResults:
    """
    Simulate a stress test scenario.

    Parameters
    ----------
    config : str
        The filepath for the stress test scenario config file.
    market_name : str
        The name of the market to simulate. TODO remove this
    num_iter : int, default 1
        The number of iterations to run.
    metrics : List[Type[Metric]], default None
        The metrics to compute.
    local : str, default ""
        The local path to a `Scenario` pickle.
    ncpu : int, default mp.cpu_count()
        The number of CPUs to use for multiprocessing.

    Returns
    -------
    MetricsResult
        An object containing the results for the simulation.
    """
    metrics = metrics or DEFAULT_METRICS

    if local:
        with open(local, "rb") as f:
            scenario_template = pickle.load(f)
    else:
        scenario_template = Scenario(config, market_name)
        scenario_template.prepare_for_run()

    metadata = {
        "scenario": config,
        "num_iter": num_iter,
        "markets": [market_name],
        "num_steps": scenario_template.num_steps,
        "freq": scenario_template.freq,
        "description": scenario_template.description,
        "template": scenario_template,
    }

    logger.info(
        "Running %d simulations with %d steps at frequency %s",
        num_iter,
        scenario_template.num_steps,
        scenario_template.freq,
    )
    strategy = BaselineStrategy(metrics)
    mcaggregator = MonteCarloProcessor(metadata)

    # TODO for other scenarios: apply shocks, etc.
    # TODO parameter sampling

    # Remove drift from collateral assets to enforce random walk
    for k, v in scenario_template.price_config["params"].items():
        if k not in STABLE_CG_IDS:
            v["mu"] = 0

    if ncpu > 1:
        logger.info("Running simulation in parallel on %d cores", ncpu)
        with multiprocessing_logging_queue() as logging_queue:
            strategy_args_list: list = [
                (deepcopy(scenario_template), {}, i + 1) for i in range(num_iter)
            ]

            wrapped_args_list = [
                (strategy, logging_queue, *args) for args in strategy_args_list
            ]

            with mp.Pool(ncpu) as pool:
                results = pool.starmap(worker, wrapped_args_list)
                pool.close()
                pool.join()

            for result in results:
                mcaggregator.collect(result)
    else:
        for i in range(num_iter):
            scenario = deepcopy(scenario_template)
            mcaggregator.collect(strategy(scenario, {}, i + 1))  # run scenario

    return mcaggregator.process()


def worker(strategy, logging_queue, *args):
    """
    Wrap strategy for multiprocess logging.
    """
    configure_multiprocess_logging(logging_queue)
    return strategy(*args)
