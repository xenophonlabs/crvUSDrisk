from metricsprocessor import MetricsProcessor
from scenario import Scenario
import logging

logging.basicConfig(
    filename="./logs/sim.log", level=logging.INFO, format="%(asctime)s %(message)s"
)


def generate(config: str):
    """
    Generate the necessary inputs, modules, and agents for a simulation.

    Parameters
    ----------
    config : str
        The filepath for the stress test scenario config file.

    Returns
    -------
    scenario : Scenario
        An object storing the necessary

    Note
    ----
    TODO
        1. Currently defaulting to params from
        ./configs/prices/1h_1694885962_1700073562.json <- 60 days of 1h data from
        Coingecko API. Should ideally allow the scenario to specify logic for
        choosing a parameter file. For example: specify that we want parameters
        learned from daily data from Coingecko for >1y of data.
        2. The initial price for simulation is the current price reported from
        Coingecko API. Include a way for user to specify a different date for
        the initial price?

    TODO should everthing here just be part of the scenario class?
    """
    logging.info()
    scenario = Scenario(config)

    # Generate inputs
    pricepaths = scenario.generate_pricepaths()

    # Generate modules
    markets = scenario.generate_markets()  # External markets

    modules = scenario.generate_crvusd_modules()

    agents = scenario.generate_agents()

    return scenario, pricepaths, markets, modules, agents


def simulate(config: str):
    """
    Simulate a stress test scenario.

    Parameters
    ----------
    config : str
        The filepath for the stress test scenario config file.

    Returns
    -------
    metrics : MetricsProcessor
        A metrics processor object, containing the
        necessary metrics for analyzing the simulation.
    """
    # Currently only simulating one LLAMMA. TODO simulate multiple LLAMMAs
    scenario, pricepaths, markets, modules, agents = generate(config)
    metrics = MetricsProcessor()

    # TODO unpack agents and modules

    for ts, prices in pricepaths:
        # Update oracles. TODO Need to actually implement the oracle
        collateral_address = llamma.metadata.collateral_address
        _p = prices[collateral_address]  # collateral/USD price
        _p_precise = int(_p * 10**18)

        # TODO update External Market price

        llamma.price_oracle_contract.set_price(_p_precise)
        llamma.prepare_for_trades(ts)  # this also updates oracle timestamps
        controller.prepare_for_trades(ts)
        # TODO prepare aggregator for trades? <- update timestamps
        # TODO prepare pegkeeper for trades? <- update timestamps
        # TODO prepare stableswap pools for trades? <- update timestamps
        # TODO prepare tricryptoo pools for trades? <- update timestamps

        # Agent actions
        # TODO add borrower action
        # TODO add liquidity provider action
        arbitrageur.do()  # arbitrages pools, liquidates positions, and updates peg keepers
        liquidator.do()
        borrower.do()  # TODO implement
        liquidity_provider.do()  # TODO implement

        # Post processing (metrics)
        metrics.update()  # TODO what should the args be?

    metrics.process()  # TODO implement

    # Notes
    # - Try tighter granularity on simulation, e.g. 1 minute?
    # - See how dependent the results are on this param.
    # - Include gas

    return metrics


def main():
    pass


if __name__ == "__main__":
    main()
