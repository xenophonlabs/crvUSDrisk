"""
Provides utils for metrics processing.
"""

from typing import Any


def entity_str(entity: Any, type_: str):
    """
    Get a simplified name for the pool
    to use in metrics column names.
    """
    if type_ == "aggregator":
        return "aggregator"
    if type_ == "stablecoin":
        return "stablecoin"
    if type_ == "agent":
        return entity.name

    if type_ == "llamma":
        name = entity.name.replace("Curve.fi Stablecoin ", "")
    elif type_ == "controller":
        name = entity.AMM.name.replace("Curve.fi Stablecoin ", "")
    elif type_ == "stableswap":
        name = entity.name.replace("Curve.fi Factory Plain Pool: ", "")
        name = name.replace("/", "_")
    elif type_ == "pk":
        name = entity.POOL.name.replace("Curve.fi Factory Plain Pool: ", "")
    else:
        raise ValueError("Invalid type_.")

    return "_".join([type_, name])
