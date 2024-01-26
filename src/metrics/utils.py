"""
Provides utils for metrics processing.
"""

from typing import Any
import numpy as np
from crvusdsim.pool.sim_interface import SimController


def entity_str(entity: Any, type_: str) -> str:
    """
    Get a simplified name for the pool
    to use in metrics column names.
    """
    if type_ == "aggregator":
        return "aggregator"
    if type_ == "stablecoin":
        return "stablecoin"
    if type_ == "agent":
        return entity.name.lower()

    if type_ == "llamma":
        name = entity.name.replace("Curve.fi Stablecoin ", "")
    elif type_ == "controller":
        name = entity.AMM.name.replace("Curve.fi Stablecoin ", "")
    elif type_ == "stableswap":
        name = entity.name.replace("Curve.fi Factory Plain Pool: ", "")
        name = name.replace("/", "_")
    elif type_ == "pk":
        name = entity.POOL.name.replace("Curve.fi Factory Plain Pool: ", "")
    elif type_ == "tricrypto":
        name = entity.name.replace("Tricrypto", "")
    else:
        raise ValueError("Invalid type_.")

    return " ".join([type_, name]).title()


def controller_healths(controller: SimController) -> np.ndarray:
    """Return array of healths of controller users."""
    return np.array(
        [controller.health(user, full=True) for user in controller.loan.keys()]
    )


def controller_debts(controller: SimController) -> np.ndarray:
    """Return array of debts of controller users."""
    return np.array(
        [
            controller._debt(user)[0]  # pylint: disable=protected-access
            for user in controller.loan
        ]
    )
