from .scenario import Scenario
import logging

__all__ = ["Scenario", "simulate"]

logging.basicConfig(
    filename="./logs/sim.log", level=logging.INFO, format="%(asctime)s %(message)s"
)


def simulate(config: str):
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
    scenario.prepare_for_run(scenario.pricepaths[0])

    for ts, sample in scenario.pricepaths:
        scenario.update_market_prices(sample)
        scenario.prepare_for_trades(ts)
        scenario.perform_actions(sample)
        scenario.after_trades()
        scenario.update_metrics()

    # TODO
    # - Try tighter granularity on simulation, e.g. 1 minute?
    # - See how dependent the results are on this param.
    # - Include gas

    return scenario.process_metrics()


def main():
    pass


if __name__ == "__main__":
    main()
