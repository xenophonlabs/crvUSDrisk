"""
Provides the function to simulate the 
baseline scenario.
"""

from typing import List, Type
from copy import deepcopy
import pickle
from .strategy import BaselineStrategy
from ...processing import MonteCarloProcessor
from ...results import MonteCarloResults
from ...scenario import Scenario
from ....logging import get_logger
from ....metrics import DEFAULT_METRICS, Metric


logger = get_logger(__name__)


def simulate(
    config: str,
    market_name: str,
    num_iter: int = 1,
    metrics: List[Type[Metric]] | None = None,
    local: str = "",
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

    Returns
    -------
    MetricsResult
        An object containing the results for the simulation.
    """

    metrics = metrics or DEFAULT_METRICS

    # TODO multiprocessing
    if local:
        with open(local, "rb") as f:
            scenario_template = pickle.load(f)
    else:
        scenario_template = Scenario(config, market_name)

    logger.info(
        "Running simulation for %d steps at frequency %s",
        scenario_template.num_steps,
        scenario_template.pricepaths.config["freq"],
    )
    strategy = BaselineStrategy(metrics)
    mcaggregator = MonteCarloProcessor()

    # TODO for other scenarios: apply shocks, etc.

    for i in range(num_iter):
        logger.info("Running iteration %d", i + 1)
        parameters: dict = {}  # TODO parameter sampling
        scenario = deepcopy(scenario_template)
        mcaggregator.collect(strategy(scenario, parameters))  # run scenario

    return mcaggregator.process()

    # TODO
    # - Try tighter granularity on simulation, e.g. 1 minute?
    # - See how dependent the results are on this param.
    # - Include gas
    # - Speed this up
    # - Include multiple LLAMMAs
    # - Include borrower actions
    # - Include LP actions
