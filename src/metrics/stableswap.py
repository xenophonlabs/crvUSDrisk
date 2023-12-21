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
        summary: Dict[str, List[str]] = {}
        plot: Dict[str, dict] = {}
        for spool in self.spools:
            spool_str = entity_str(spool, "stableswap")
            summary[spool_str + "_price"] = []
            summary[spool_str + "_ma_price"] = []
            summary[spool_str + "_lp_supply"] = []
            summary[spool_str + "_virtual_price"] = ["max"]
            plot[spool_str + "_virtual_price"] = {
                "title": f"{spool_str} Virtual Price",
                "kind": "line",
            }
            for symbol in spool.assets.symbols:
                summary["_".join([spool_str, symbol, "bal"])] = []
        return {"functions": {"summary": summary}, "plot": plot}

    def compute(self) -> Dict[str, Union[int, float]]:
        res = []
        for spool in self.spools:
            i = get_crvusd_index(spool)
            res.append(spool.price(i ^ 1, i) / 1e18)
            res.append(spool.price_oracle() / 1e18)
            res.append(spool.totalSupply / 1e18)
            res.append(spool.get_virtual_price() / 1e18)
            for i in range(len(spool.assets.symbols)):
                res.append(spool.balances[i] / 1e18)
        return dict(zip(self.cols, res))
