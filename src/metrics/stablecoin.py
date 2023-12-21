"""
Provides metrics on the Stablecoin.
"""

from typing import TYPE_CHECKING, Union, Dict, List
from functools import cached_property
from .base import Metric
from .utils import entity_str

if TYPE_CHECKING:
    from crvusdsim.pool import StableCoin


class StablecoinMetrics(Metric):
    """
    Metrics computed on the Stablecoin.
    """

    def __init__(self, **kwargs) -> None:
        self.stablecoin: StableCoin = kwargs["stablecoin"]

    @cached_property
    def config(self) -> dict:
        summary: Dict[str, List[str]] = {}
        plot: Dict[str, dict] = {}
        stablecoin_str = entity_str(self.stablecoin, "stablecoin")
        summary[stablecoin_str + "_total_supply"] = ["mean"]
        plot[stablecoin_str + "_total_supply"] = {
            "title": "Stablecoin Total Supply",
            "kind": "line",
        }
        return {"functions": {"summary": summary}, "plot": plot}

    def compute(self) -> Dict[str, Union[int, float]]:
        res = []
        res.append(self.stablecoin.totalSupply / 1e18)
        return dict(zip(self.cols, res))
