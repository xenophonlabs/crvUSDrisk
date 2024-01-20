"""
Tools for running simulation scenarios, inspired by 
https://github.com/curveresearch/curvesim/blob/main/curvesim/pipelines/__init__.py.

The basic model for a scenario is to:
1. Generate a `Scenario` object to be used as a template.
    This includes Curve assets (like LLAMMAs, Controllers,
    Stableswap pools), as well as External Markets, and Agents 
    (e.g. Liquidator, Arbitrageur).
2. Apply scenario shocks required for the scenario. For example, the "baseline"
    scenario will apply enforce 0-drift on collateral GBMs, whereas the "high volatility" 
    scenario will increase the volatility parameter.
3. Run the simulation. The `Scenario` template is copied into multiple parallel
    processes that each run their own simulation.
4. Aggregate results. The results from each simulation are aggregated together
    and plots/tables are generated to display statistically significant results.
"""
import multiprocessing as mp
from typing import List, Type, Any, Dict
from copy import deepcopy
from queue import Queue
from .strategy import Strategy
from .processing import MonteCarloProcessor
from .results import MonteCarloResults, SingleSimResults
from .scenario import Scenario
from ..logging import (
    get_logger,
    multiprocessing_logging_queue,
    configure_multiprocess_logging,
)
from ..metrics import DEFAULT_METRICS, Metric
from ..configs.parameters import MODELED_PARAMETERS


logger = get_logger(__name__)


# pylint: disable=too-many-arguments, too-many-locals
def simulate(
    config: str,
    market_names: List[str],
    num_iter: int = 1,
    metrics: List[Type[Metric]] | None = None,
    ncpu: int = mp.cpu_count(),
    to_sweep: List[Dict[str, Any]] | None = None,
) -> List[MonteCarloResults]:
    """
    Run a simulation scenario for given parameters.

    Parameters
    ----------
    config : str
        The name of the scenario to simulate. Must be defined in
        `src/configs/scenarios.py`.
    market_names : List[str]
        The names of the markets to simulate.
    num_iter : int, default 1
        The number of iterations to run.
    metrics : List[Type[Metric]], default None
        The metrics to compute.
    ncpu : int, default mp.cpu_count()
        The number of CPUs to use for multiprocessing.
    to_sweep: List[Dict[str, Any]], default None
        List of parameters to sweep.

    Returns
    -------
    List[MetricsResult]
        A list of objects containing simulation results for each parameter set.
    """
    metrics = metrics or DEFAULT_METRICS
    mcresults = []
    to_sweep = to_sweep or [{}]

    template = Scenario(config, market_names)  # shocks applied here

    for params in to_sweep:
        scenario_template = deepcopy(template)
        scenario_template.apply_parameters(params)

        if params:
            for k in params:
                assert k in MODELED_PARAMETERS, f"Invalid parameter type: {k}"
            logger.info("Trying parameter set: %s", params)

        metadata = {
            "scenario": config,
            "num_iter": num_iter,
            "markets": market_names,
            "num_steps": scenario_template.num_steps,
            "freq": scenario_template.freq,
            "template": scenario_template,
            "params": params,
        }

        logger.info(
            "Running scenario %s with %d iterations with %d steps at frequency %s.",
            config,
            num_iter,
            scenario_template.num_steps,
            scenario_template.freq,
        )

        strategy = Strategy(metrics)
        mcaggregator = MonteCarloProcessor(metadata)

        if ncpu > 1:
            mp.set_start_method("spawn", force=True)  # safer than `fork` on Unix
            logger.info("Running simulation in parallel on %d cores", ncpu)
            with multiprocessing_logging_queue() as logging_queue:
                strategy_args_list: list = [
                    (deepcopy(scenario_template), i + 1) for i in range(num_iter)
                ]

                wrapped_args_list = [
                    (strategy, logging_queue, *args) for args in strategy_args_list
                ]

                with mp.Pool(ncpu) as pool:
                    results = pool.starmap(worker, wrapped_args_list)
                    pool.close()
                    pool.join()

                for result in results:
                    if result:
                        mcaggregator.collect(result)
        else:
            for i in range(num_iter):
                scenario = deepcopy(scenario_template)
                mcaggregator.collect(strategy(scenario, i + 1))

        mcresult = mcaggregator.process()
        # Force computation of summary to cache it
        mcresult.summary  # pylint: disable=pointless-statement

        mcresults.append(mcresult)

    return mcresults


def worker(
    strategy: Strategy,
    logging_queue: Queue,
    scenario: Scenario,
    i: int,
) -> SingleSimResults | None:
    """
    Wrap strategy for multiprocess logging.
    """
    configure_multiprocess_logging(logging_queue)
    try:
        result = strategy(scenario, i)
    except AssertionError as e:
        logger.critical("Failed run %d with exception %s", i, e)
        return None
    return result
