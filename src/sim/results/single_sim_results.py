"""
Provides the `SingleSimResults` dataclass.
"""

from typing import List
from dataclasses import dataclass
from functools import cached_property
import pandas as pd
import matplotlib.pyplot as plt
from ...metrics import Metric
from ...prices import PricePaths
from ...plotting.sim import plot_prices
from ...plotting.utils import make_square

FIGSIZE = (10, 10)


@dataclass
class SingleSimResults:  # pylint: disable=too-few-public-methods
    """
    Stores metrics data for a single simulation
    """

    def __init__(
        self, df: pd.DataFrame, pricepaths: PricePaths, metrics: List[Metric]
    ) -> None:
        self.df = df.set_index("timestamp")
        self.pricepaths = pricepaths
        self.metrics = metrics

    @property
    def metric_map(self):
        """Return the metric map."""
        return {type(metric).__name__: i for i, metric in enumerate(self.metrics)}

    def plot_metric(self, metric_index: int, show: bool = True):
        """Plot metric."""
        # TODO pass in axs
        metric = self.metrics[metric_index]
        plot_config = metric.config["plot"]
        cols = plot_config.keys()
        _df = self.df[cols]
        n, m = make_square(len(cols))
        _df.plot(
            subplots=True,
            layout=(n, m),
            legend=False,
            figsize=FIGSIZE,
            title=[x["title"] for x in list(plot_config.values())],
            sharex=False,
            rot=45,
        )
        if show:
            plt.tight_layout()
            plt.show()

    def plot_prices(self):
        """Plot the prices."""
        plot_prices(self.pricepaths.prices)

    @cached_property
    def agg_config(self):
        """
        Config for aggregating metrics.
        """
        agg = {}
        for metric in self.metrics:
            cfg = metric.config["functions"]["summary"]
            for col, funcs in cfg.items():
                if funcs:
                    funcs = [f if f != "last" else "max" for f in funcs]
                    agg.update({col: funcs})
        return agg

    def summarize(self) -> pd.DataFrame:
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
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        return summary
