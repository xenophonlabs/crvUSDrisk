"""
Utility functions.
"""
from __future__ import annotations
from typing import TYPE_CHECKING
import requests as req
import pandas as pd

if TYPE_CHECKING:
    from ..types import SimPoolType

QUOTES_URL = "http://97.107.138.106/quotes"
TIMEOUT = 60


def get_crvusd_index(pool: SimPoolType) -> int:
    """
    Return index of crvusd in pool.
    """
    symbols = [c.symbol for c in pool.coins]
    return symbols.index("crvUSD")


def get_quotes(start: int, end: int) -> pd.DataFrame:
    """
    Return list of 1inch quotes from API.
    TODO add support for filtering for tokens.
    """
    params = {
        "start": start,
        "end": end,
    }
    res = req.get(QUOTES_URL, params=params, timeout=TIMEOUT)
    res.raise_for_status()
    quotes = pd.DataFrame(res.json()).set_index(["src", "dst"])
    return quotes
