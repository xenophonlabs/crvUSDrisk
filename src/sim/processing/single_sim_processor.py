"""
Provides the `SingleSimProcessor` class.

The `SingleSimProcessor` is inspired by the `StateLog` class in
`curvesim`. The SSP will log lower-dimension metrics of each entity
in the simulation (LLAMMAs, Controllers, Agents, etc.) to avoid
storing the entire state of the simulation at each timestep (which
would create large memory consumption with many timesteps in a 
multiprocessing env).

The SSP will `update` at each timestep, computing each metric and 
storing them in a DataFrame. At the end of each simulation, these
metrics are processed into a `SingleSimResults` object and collected
by the `MonteCarloProcessor`.
"""

from typing import List, Dict, Union, Type
import pandas as pd
from ..results import SingleSimResults
from ..scenario import Scenario
from ...metrics import Metric, init_metrics


class SingleSimProcessor:
    """
    Stores and processes metrics data
    for a single simulation.

    TODO narrow down to the useful ones.
    """

    def __init__(self, scenario: Scenario, metrics: List[Type[Metric]]):
        # Unpack scenario into kwargs
        kwargs = {
            "agents": scenario.agents,
            "aggregator": scenario.aggregator,
            "controllers": [
                scenario.controller
            ],  # TODO incorporate multiple controllers
            "llammas": [scenario.llamma],  # TODO incorporate multiple LLAMMAs
            "pegkeepers": scenario.peg_keepers,
            "stablecoin": scenario.stablecoin,
            "spools": scenario.stableswap_pools,
        }

        self.metrics = init_metrics(metrics, **kwargs)

        cols: List[str] = ["timestamp"]

        for metric in self.metrics:
            cols.extend(metric.cols)  # type: ignore

        self.cols = cols
        self.results: pd.DataFrame = pd.DataFrame(columns=self.cols)

        self.pricepaths = scenario.pricepaths

        # Initial state
        self.initial_state = self.update(scenario.pricepaths[0].timestamp)

    def update(self, ts: int, inplace: bool = False) -> Dict[str, Union[float, int]]:
        """
        Collect metrics for the current timestep of the sim.
        """
        res: Dict[str, Union[float, int]] = {"timestamp": ts}

        for metric in self.metrics:
            res.update(metric.compute())

        if inplace:
            self.results.loc[len(self.results)] = res

        return res

    def process(self) -> SingleSimResults:
        """Process timeseries df into metrics result."""
        return SingleSimResults(self.results, self.pricepaths, self.metrics)
