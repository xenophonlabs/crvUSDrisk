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
from collections import defaultdict
import pandas as pd
from ..results import SingleSimResults
from ..scenario import Scenario
from ...metrics import Metric, init_metrics
from ...metrics.utils import controller_debts, controller_healths


class SingleSimProcessor:
    """
    Stores and processes metrics data
    for a single simulation.
    """

    # pylint: disable=too-many-instance-attributes
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

        self.initial_debts = self.controller_debts()
        self.debt_by_band = self.get_debt_by_band()

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

    def metric_kwargs(self) -> dict:
        """
        Get kwargs for all metrics. This prevents
        us from repeating expensive operations like
        calculating healths.
        """
        kwargs: Dict[str, Any] = {
            "healths": {},
            "debts": {},
            "initial_debts": self.initial_debts,
            "total_initial_debt": sum(self.initial_debts.values()),
        }

        for controller in self.scenario.controllers:
            kwargs["healths"][controller.address] = controller_healths(controller)
            kwargs["debts"][controller.address] = controller_debts(controller)

        return kwargs

    def controller_debts(self) -> Dict[str, float]:
        """Initial debt of each controller."""
        return {
            controller.address: controller.total_debt() / 1e18
            for controller in self.scenario.controllers
        }

    def get_debt_by_band(self) -> Dict[str, Dict[int, float]]:
        """
        Get the debt by band for each controller.
        """
        debt_by_band: Dict[str, Dict[int, float]] = {}
        for controller in self.scenario.controllers:
            debt_by_band[controller.address] = defaultdict(float)
            llamma = controller.AMM
            for user in controller.loan:
                debt = controller.debt(user) / 1e18
                n1, n2 = llamma.read_user_tick_numbers(user)
                n = n2 - n1 + 1
                for band in range(n1, n2 + 1):
                    debt_by_band[controller.address][band] += debt / n
        return debt_by_band

    def active_debt(self) -> Dict[str, float]:
        """
        Get the
        """
        active_bands = defaultdict(set)
        for controller in self.scenario.controllers:
            llamma = controller.AMM
            for band, val in llamma.bands_fees_x.items():
                if val > 0:
                    active_bands[controller.address].add(band)
            for band, val in llamma.bands_fees_y.items():
                if val > 0:
                    active_bands[controller.address].add(band)
        print(active_bands)
        active_debt: Dict[str, float] = defaultdict(int)
        for controller in self.scenario.controllers:
            for band in active_bands[controller.address]:
                active_debt[controller.address] += self.debt_by_band[
                    controller.address
                ][band]
        return active_debt

    def process(self) -> SingleSimResults:
        """Process timeseries df into metrics result."""
        active_debt = self.active_debt()
        self.prune()
        return SingleSimResults(
            self.results,
            self.pricepaths,
            self.metrics,
            self.initial_debts,
            active_debt,
        )

    def prune(self) -> None:
        """Prune size."""
        del self.scenario
        del self.oracles
