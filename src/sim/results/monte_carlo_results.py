"""
Provides the `MonteCarloResults` dataclass.
"""
from __future__ import annotations
from typing import List, TYPE_CHECKING
from dataclasses import dataclass
from functools import cached_property
import pandas as pd
import matplotlib.pyplot as plt
from ...plotting.utils import make_square

if TYPE_CHECKING:
    from .single_sim_results import SingleSimResults


FIGSIZE = (20, 20)


@dataclass
class MonteCarloResults:  # pylint: disable=too-few-public-methods
    """
    Stores metrics data aggregated over many simulations.
    """

    data: List[SingleSimResults]
    metadata: dict | None = None

    @property
    def metric_map(self):
        """Map metric ids to names."""
        return self.data[0].metric_map

    def plot_runs(self, metric_id: int):
        """Plot metric for each run."""
        axs = None
        for i, _df in enumerate(self.data):
            show = False
            if i == len(self.data) - 1:
                show = True
            axs = _df.plot_metric(metric_id, axs=axs, show=show)

    def plot_summary(self, cols: List[str] | None = None, show: bool = True):
        """Plot histogram of summary metrics."""
        cols = cols or self.key_agg_cols

        n, m = make_square(len(cols))

        f, axs = plt.subplots(n, m, figsize=FIGSIZE)

        if n == 1 and m == 1:
            axs.hist(self.summary[cols[0]], bins=self.summary.shape[0])
            axs.set_title(cols[0])
        elif n == 1:
            for i in range(m):
                axs[i].hist(self.summary[cols[i]], bins=self.summary.shape[0])
                axs[i].set_title(cols[i])
        else:
            for i in range(n):
                for j in range(m):
                    if len(cols):
                        col = cols.pop(0)
                        axs[i, j].hist(self.summary[col], bins=self.summary.shape[0])
                        axs[i, j].set_title(col)
                    else:
                        break

        if show:
            f.tight_layout()
            plt.show()

    @cached_property
    def key_agg_cols(self):
        """Key columns."""
        return self.data[0].key_agg_cols

    @cached_property
    def summary(self):
        """Summarize metrics."""
        summary = pd.concat([x.summary for x in self.data])
        summary.index = range(len(summary))
        summary.index.name = "Run ID"
        return summary
