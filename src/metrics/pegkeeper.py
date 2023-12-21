"""
Provides metrics on the PegKeepers.
"""

from typing import List, TYPE_CHECKING, Union, Dict
from functools import cached_property
from .base import Metric
from .utils import entity_str

if TYPE_CHECKING:
    from crvusdsim.pool import PegKeeper


class PegKeeperMetrics(Metric):
    """
    Metrics computed on StableSwap pools.
    """

    def __init__(self, **kwargs) -> None:
        self.pks: List[PegKeeper] = kwargs["pegkeepers"]

    @cached_property
    def config(self) -> dict:
        summary: Dict[str, List[str]] = {}
        plot: Dict[str, dict] = {}
        for pk in self.pks:
            pk_str = entity_str(pk, "pk")
            summary[pk_str + "_debt"] = ["mean", "max"]
            plot[pk_str + "_debt"] = {
                "title": f"{pk_str} Debt",
                "kind": "line",
            }
            summary[pk_str + "_profit"] = ["max"]  # proxy for `last`
            plot[pk_str + "_profit"] = {
                "title": f"{pk_str} Profit",
                "kind": "line",
            }
        return {"functions": {"summary": summary}, "plot": plot}

    def compute(self) -> Dict[str, Union[int, float]]:
        res = []
        for pk in self.pks:
            res.append(pk.debt / 1e18)
            res.append(pk.calc_profit())
        return dict(zip(self.cols, res))
