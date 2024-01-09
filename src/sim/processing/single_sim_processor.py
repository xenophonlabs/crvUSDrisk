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

from typing import List, Dict, Union, Type, Any
import pandas as pd
from ..results import SingleSimResults
from ..scenario import Scenario
from ...metrics import Metric, init_metrics
from ...metrics.utils import controller_debts, controller_healths


class SingleSimProcessor:
    """
    Stores and processes metrics data
    for a single simulation.

    TODO narrow down to the useful ones.
    """

    def __init__(self, scenario: Scenario, metrics: List[Type[Metric]]) -> None:
        self.metrics = init_metrics(metrics, scenario)

        cols: List[str] = ["timestamp"]

        for metric in self.metrics:
            cols.extend(metric.cols)  # type: ignore

        self.cols = cols
        self.results: pd.DataFrame = pd.DataFrame(columns=self.cols)

        self.pricepaths = scenario.pricepaths
        self.oracles = scenario.oracles
        self.scenario = scenario

        # Initial state
        self.initial_state = self.update(scenario.pricepaths[0].timestamp)

    def update(self, ts: int, inplace: bool = False) -> Dict[str, Union[float, int]]:
        """
        Collect metrics for the current timestep of the sim.
        """

        for oracle in self.oracles:
            oracle.freeze()

        res: Dict[str, Union[float, int]] = {"timestamp": ts}
        kwargs = self.metric_kwargs()

        for metric in self.metrics:
            res.update(metric.compute(**kwargs))

        if inplace:
            self.results.loc[len(self.results)] = res

        for oracle in self.oracles:
            oracle.unfreeze()

        return res

    def metric_kwargs(self) -> Dict[str, Any]:
        """
        Get kwargs for all metrics. This prevents
        us from repeating expensive operations like
        calculating healths.
        """
        kwargs: Dict[str, Any] = {
            "healths": {},
            "debts": {},
        }

        for controller in self.scenario.controllers:
            kwargs["healths"][controller.address] = controller_healths(controller)
            kwargs["debts"][controller.address] = controller_debts(controller)

        return kwargs

    def process(self) -> SingleSimResults:
        """Process timeseries df into metrics result."""
        self.prune()
        return SingleSimResults(self.results, self.pricepaths, self.metrics)

    def prune(self) -> None:
        """Prune size."""
        del self.scenario
        del self.oracles
