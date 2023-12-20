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
        summary: Dict[str, str] = {}
        for llamma in self.llammas:
            # TODO summary statistics methodology
            llamma_str = entity_str(llamma, "llamma")
            summary[llamma_str + "_price"] = "mean"
            summary[llamma_str + "_oracle_price"] = "mean"
            summary[llamma_str + "_fees_x"] = "mean"
            summary[llamma_str + "_fees_y"] = "mean"
            summary[llamma_str + "_bal_x"] = "mean"
            summary[llamma_str + "_bal_y"] = "mean"
        # TODO plot config
        return {"functions": {"summary": summary}}

    def compute(self) -> Dict[str, Union[int, float]]:
        res = []
        for llamma in self.llammas:
            res.append(llamma.price(0, 1))
            res.append(llamma.price_oracle())
            res.append(llamma.admin_fees_x)
            res.append(llamma.admin_fees_y)
            res.append(sum(llamma.bands_x.values()))
            res.append(sum(llamma.bands_y.values()))
        return dict(zip(self.cols, res))
