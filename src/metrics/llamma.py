"""
Provides metrics on the LLAMMAs.
"""

from typing import List, TYPE_CHECKING, Union, Dict
from functools import cached_property
from .base import Metric
from .utils import entity_str

if TYPE_CHECKING:
    from crvusdsim.pool.sim_interface import SimLLAMMAPool


class LLAMMAMetrics(Metric):
    """
    Metrics computed on LLAMMAs.
    """

    def __init__(self, **kwargs) -> None:
        self.llammas: List[SimLLAMMAPool] = kwargs["llammas"]

    @cached_property
    def config(self) -> dict:
        summary: Dict[str, List[str]] = {}
        plot: Dict[str, dict] = {}
        for llamma in self.llammas:
            llamma_str = entity_str(llamma, "llamma")
            summary[llamma_str + "_price"] = []
            summary[llamma_str + "_oracle_price"] = []
            summary[llamma_str + "_fees_x"] = ["sum"]
            plot[llamma_str + "_fees_x"] = {
                "title": f"{llamma_str} Fees X",
                "kind": "line",
            }
            summary[llamma_str + "_fees_y"] = ["sum"]
            plot[llamma_str + "_fees_y"] = {
                "title": f"{llamma_str} Fees Y",
                "kind": "line",
            }
            summary[llamma_str + "_bal_x"] = []
            summary[llamma_str + "_bal_y"] = []
        return {"functions": {"summary": summary}, "plot": plot}

    def compute(self) -> Dict[str, Union[int, float]]:
        res = []
        for llamma in self.llammas:
            res.append(llamma.price(0, 1) / 1e18)
            res.append(llamma.price_oracle() / 1e18)
            res.append(llamma.admin_fees_x)
            res.append(llamma.admin_fees_y)
            res.append(sum(llamma.bands_x.values()) / 1e18)
            res.append(sum(llamma.bands_y.values()) / 1e18)
        return dict(zip(self.cols, res))

    def prune(self) -> None:
        del self.llammas
