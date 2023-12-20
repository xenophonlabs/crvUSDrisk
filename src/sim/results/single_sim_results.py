"""
Provides the `SingleSimResults` dataclass.
"""

from typing import List
from dataclasses import dataclass
import pandas as pd
from ...metrics import Metric


@dataclass
class SingleSimResults:  # pylint: disable=too-few-public-methods
    """
    Stores metrics data for a single simulation
    """

    df: pd.DataFrame
    metrics: List[Metric]

    def plot(self):
        """Plot metrics."""

    def summarize(self):
        """Summarize metrics."""
