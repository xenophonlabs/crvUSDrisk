"""
Tools for running simulation scenarios, inspired by 
https://github.com/curveresearch/curvesim/blob/main/curvesim/pipelines/__init__.py.

The basic model for a scenario is to:
1. Generate a `Scenario` object to be used as a template.
    This includes Curve assets (like LLAMMAs, Controllers,
    Stableswap pools), as well as External Markets, and Agents 
    (e.g. Liquidator, Arbitrageur).
2. Apply the pre-processing required for the `Scenario`. For example, the "baseline_micro"
    scenario will apply basic pre-processing (equilibrate pool prices via
    an initial arbitrage), whereas the "liquidity crunch" scneario will
    stochastically remove liquidity from the system.
3. Generate prices for the `Scenario`. For example, the "baseline_micro" scenario
    will generate prices based on recent historical price data using GBMs and OU
    processes. The "bear" scenario will depress each token's "drift" parameter,
    whereas the "high volatility" scenario will augment the "volatility" parameter,
    and the "flash crash" scenario will apply negative jumps.
4. Run the simulation. The `Scenario` template is copied into multiple parallel
    processes that each run their own simulation. 
5. Aggregate results. The results from each simulation are aggregated together
    and plots/tables are generated to display statistically significant results.
"""
import multiprocessing as mp
from ..logging import get_logger
from .scenarios import SCENARIO_MAP
from .results import MonteCarloResults

logger = get_logger(__name__)


def run_scenario(
    scenario_name: str,
    market_name: str,
    num_iter: int = 1,
    local: str = "",
    ncpu: int = mp.cpu_count(),
) -> MonteCarloResults:
    """
    Core function for running simulation scenarios.
    """
    # TODO consider multiple markets
    func = SCENARIO_MAP[scenario_name]
    logger.info("Running scenario: %s", scenario_name)
    output = func(scenario_name, market_name, num_iter=num_iter, local=local, ncpu=ncpu)
    logger.info("Completed scenario: %s", scenario_name)
    return output
