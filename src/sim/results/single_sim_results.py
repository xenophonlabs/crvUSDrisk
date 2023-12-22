"""
Provides the `SingleSimResults` dataclass.
"""

from typing import List, Any
from dataclasses import dataclass
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

    def plot_metric(self, metric_index: int, axs: Any = None, show: bool = True):
        """Plot metric."""
        metric = self.metrics[metric_index]
        plot_config = metric.config["plot"]
        cols = list(plot_config.keys())
        titles = [x["title"] for x in list(plot_config.values())]
        n, m = make_square(len(cols))

        if axs is None:
            _, axs = plt.subplots(n, m, figsize=FIGSIZE)

        if n == 1 and m == 1:
            axs.plot(self.df[cols[0]])
            axs.set_title(titles[0])
            axs.tick_params(axis="x", rotation=45)
        elif n == 1:
            for i in range(m):
                axs[i].plot(self.df[cols[i]])
                axs[i].set_title(titles[i])
                axs[i].tick_params(axis="x", rotation=45)
        else:
            # TODO refactor this
            # pylint: disable=duplicate-code
            for i in range(n):
                for j in range(m):
                    if cols:
                        col = cols.pop(0)
                        title = titles.pop(0)
                        axs[i, j].plot(self.df[col])
                        axs[i, j].set_title(title)
                        axs[i, j].tick_params(axis="x", rotation=45)
                    else:
                        break

        if show:
            plt.tight_layout()
            plt.show()

        return axs

    def plot_prices(self):
        """Plot the prices."""
        plot_prices(self.pricepaths.prices)

    @property
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
        summary.columns = ["_".join(col).strip() for col in summary.columns.values]
        return summary
