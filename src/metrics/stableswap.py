"""
Provides metrics on the LLAMMAs.
"""

from typing import List, TYPE_CHECKING, Union, Dict
from functools import cached_property
from .base import Metric
from .utils import entity_str
from ..utils import get_crvusd_index

if TYPE_CHECKING:
    from crvusdsim.pool.sim_interface import SimCurveStableSwapPool


class StableSwapMetrics(Metric):
    """
    Metrics computed on StableSwap pools.
    """

    def __init__(self, **kwargs) -> None:
        self.spools: List[SimCurveStableSwapPool] = kwargs["spools"]

    @cached_property
    def config(self) -> dict:
        summary: Dict[str, str] = {}
        for spool in self.spools:
            # TODO summary statistics methodology
            spool_str = entity_str(spool, "stableswap")
            summary[spool_str + "_price"] = "mean"
            summary[spool_str + "_ma_price"] = "mean"
            summary[spool_str + "_lp_supply"] = "mean"
            summary[spool_str + "_virtual_price"] = "mean"
            for symbol in spool.assets.symbols:
                summary["_".join([spool_str, symbol, "bal"])] = "mean"
        # TODO plot config
        return {"functions": {"summary": summary}}

    def compute(self) -> Dict[str, Union[int, float]]:
        res = []
        for spool in self.spools:
            i = get_crvusd_index(spool)
            res.append(spool.price(i ^ 1, i))
            res.append(spool.price_oracle())
            res.append(spool.totalSupply)
            res.append(spool.get_virtual_price())
            for i in range(len(spool.assets.symbols)):
                res.append(spool.balances[i])
        return dict(zip(self.cols, res))
