"""
Provides the `SingleSimResults` dataclass.
"""

from typing import List, Any, Dict
from dataclasses import dataclass
import pandas as pd
import matplotlib.pyplot as plt
from ...metrics import Metric
from ...prices import PricePaths
from ...plotting.sim import plot_prices

FIGSIZE = (10, 10)


@dataclass
class SingleSimResults:
    """
    Stores metrics data for a single simulation
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        df: pd.DataFrame,
        pricepaths: PricePaths,
        metrics: List[Metric],
        initial_debt: Dict[str, float],
        active_debt: Dict[str, float],
    ) -> None:
        self.df = df.set_index("timestamp")
        self.pricepaths = pricepaths
        self.metrics = metrics
        self.initial_debt = initial_debt
        self.active_debt = active_debt

    @property
    def metric_map(self) -> Dict[str, int]:
        """Return the metric map."""
        return {type(metric).__name__: i for i, metric in enumerate(self.metrics)}

    @property
    def pct_debt_active(self) -> Dict[str, float]:
        """Return the percentage of debt active."""
        return {
            k: v / self.initial_debt[k] * 100
            for k, v in self.active_debt.items()
            if v != 0
        }

    def plot_metric(
        self, metric_index: int, axs: Any = None, show: bool = True, i: int = -1
    ) -> plt.Axes:
        """
        Plot the ith metric of `self.metrics[metric_index]`.
        If `i` is -1, plot the key metric.
        """
        metric = self.metrics[metric_index]

        if i == -1:
            submetric = metric.key_metric
        else:
            submetric = list(metric.config.keys())[i]

        if not axs:
            _, axs = plt.subplots(figsize=FIGSIZE)

        axs.plot(self.df[submetric])
        axs.set_title(submetric)
        axs.tick_params(axis="x", rotation=45)

        if show:
            plt.tight_layout()
            plt.show()

        return axs

    def plot_prices(self, show: bool = True) -> None:
        """Plot the prices."""
        plot_prices(self.pricepaths.prices)
        if show:
            plt.show()

    @property
    def agg_config(self) -> Dict[str, List[str]]:
        """
        Config for aggregating metrics.
        """
        agg = {}
        for metric in self.metrics:
            cfg = metric.config
            for col, funcs in cfg.items():
                if funcs:
                    agg.update({col: funcs})
        return agg

    @property
    def cols(self) -> List[str]:
        """Column names."""
        return list(self.agg_config.keys())

    @property
    def key_metrics(self) -> List[str]:
        """Key metrics."""
        return [m.key_metric for m in self.metrics]

    @property
    def key_agg_cols(self) -> List[str]:
        """Key agg columns."""
        cols = []
        for metric in self.key_metrics:
            cols.extend(
                [" ".join([metric, func]).title() for func in self.agg_config[metric]]
            )
        return cols

    @property
    def summary(self) -> pd.DataFrame:
        """Summarize metrics."""
        summary = self.df.agg(self.agg_config)
        summary = (
            summary.reset_index()
            .pivot(columns=["index"], values=summary.columns)
            .dropna(axis=1, how="all")
            .apply(lambda x: x[x.first_valid_index()])
            .to_frame()
            .T
        )
        summary.columns = [
            " ".join(col).strip().title() for col in summary.columns.values
        ]
        return summary
