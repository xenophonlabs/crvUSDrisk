"""Useful type aliases."""
from typing import TypeAlias, Union
from crvusdsim.pool.sim_interface import SimLLAMMAPool, SimCurveStableSwapPool
from curvesim.pool.sim_interface import SimCurvePool
from ..modules import ExternalMarket

SimPoolType: TypeAlias = Union[
    SimLLAMMAPool, SimCurveStableSwapPool, SimCurvePool, ExternalMarket
]
