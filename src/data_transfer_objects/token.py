"""Simple DTO for token data."""
from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class TokenDTO:
    # TODO DTO min/max trade sizes should be dynamic
    address: str
    name: str
    symbol: str
    decimals: int
    min_trade_size: int  # min amt_in for 1inch quotes
    max_trade_size: int  # max amt_in for 1inch quotes