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
        summary: Dict[str, str] = {}
        for pk in self.pks:
            # TODO summary statistics methodology
            pk_str = entity_str(pk, "pk")
            summary[pk_str + "_debt"] = "mean"
            summary[pk_str + "_profit"] = "mean"
        # TODO plot config
        return {"functions": {"summary": summary}}

    def compute(self) -> Dict[str, Union[int, float]]:
        res = []
        for pk in self.pks:
            res.append(pk.debt)
            res.append(pk.calc_profit())
        return dict(zip(self.cols, res))
