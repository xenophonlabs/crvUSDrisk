"""
Main module for the simulation package. Runs a stress test scenario for a 
given stress test configuration.
"""
import pickle
from .scenario import Scenario
from ..metrics import MetricsProcessor
from ..plotting.sim import plot_prices
from ..logging import get_logger

__all__ = ["Scenario", "sim"]

logger = get_logger(__name__)


def sim(config: str):
    """
    Simulate a stress test scenario.

    Parameters
    ----------
    config : str
        The filepath for the stress test scenario config file.

    Returns
    -------
    MetricsResult
        An object containing the results for the simulation.
    """
    scenario = Scenario(config)
    scenario.prepare_for_run()
    logger.info(
        "Running simulation for %d steps at frequency %s",
        scenario.num_steps,
        scenario.pricepaths.config["freq"],
    )
    _ = plot_prices(scenario.pricepaths.prices, fn="./figs/sims/prices.png")
    metrics_processor = MetricsProcessor(scenario)

    for sample in scenario.pricepaths:
        scenario.prepare_for_trades(sample)
        scenario.perform_actions(sample)
        # scenario.after_trades()
        metrics_processor.update()

    # Results are dumped into metrics_processor!
    with open("./pickles/scenario.pkl", "wb") as f:
        pickle.dump(scenario, f)
    with open("./pickles/metrics_processor.pkl", "wb") as f:
        pickle.dump(metrics_processor, f)

    # TODO
    # - Try tighter granularity on simulation, e.g. 1 minute?
    # - See how dependent the results are on this param.
    # - Include gas
    # - Speed this up
    # - Include multiple LLAMMAs
    # - Include borrower actions
    # - Include LP actions

    # return metrics_processor.process()
