"""
Main module for the simulation package. Runs a stress test scenario for a 
given stress test configuration.
"""
import logging
from .scenario import Scenario
from ..metrics import MetricsProcessor

__all__ = ["Scenario", "sim"]

logging.basicConfig(
    filename="./logs/sim.log", level=logging.INFO, format="%(asctime)s %(message)s"
)


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
    # Currently only simulating one LLAMMA. TODO simulate multiple LLAMMAs
    scenario = Scenario(config)
    scenario.prepare_for_run()
    metrics_processor = MetricsProcessor(scenario)

    for _, sample in scenario.pricepaths:
        scenario.update_market_prices(sample)
        scenario.prepare_for_trades(sample)
        scenario.perform_actions(sample)
        scenario.after_trades()
        metrics_processor.update()

    # TODO
    # - Try tighter granularity on simulation, e.g. 1 minute?
    # - See how dependent the results are on this param.
    # - Include gas

    return metrics_processor.process()
