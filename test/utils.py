"""
Provides testing utilities.
"""
from typing import List
from crvusdsim.pool.crvusd.utils import BlocktimestampMixins


def increment_timestamps(objs: List[BlocktimestampMixins], td: int = 60 * 60) -> None:
    """
    Increment the timestep for all the input objects.
    """
    ts = objs[0]._block_timestamp + td
    for obj in objs:
        obj._block_timestamp = ts


def approx(x1: int | float, x2: int | float, tol: float = 1e-3) -> bool:
    """
    Check that abs(x1 - x2)/x1 <= tol.
    """
    return abs(x1 - x2) / x1 <= tol
