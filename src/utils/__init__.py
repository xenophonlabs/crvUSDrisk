"""
Utility functions.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Set, Dict, List
import asyncio
from collections import defaultdict
import requests as req
import pandas as pd
import numpy as np
from crvusdsim.network.subgraph import get_user_snapshots, _stableswap_snapshot
from crvusdsim.pool import (
    StableCoin,
    CurveStableSwapPoolMetaData,
    CurveStableSwapPool,
    SimCurveStableSwapPool,
)
from curvesim.network.utils import sync

if TYPE_CHECKING:
    from ..data_transfer_objects import TokenDTO
    from ..types import SimPoolType

QUOTES_URL = "http://97.107.138.106/quotes"
TIMEOUT = 60


def get_crvusd_index(pool: SimPoolType) -> int:
    """
    Return index of crvusd in pool.
    """
    symbols = [c.symbol for c in pool.coins]
    return symbols.index("crvUSD")


def get_quotes(start: int, end: int, tokens: Set[TokenDTO]) -> pd.DataFrame:
    """
    Return list of 1inch quotes from API.
    TODO add support for filtering for tokens.
    """
    tokens_raw = ",".join([t.address for t in tokens])
    params: Dict[str, int | str] = {
        "start": start,
        "end": end,
        "tokens": tokens_raw,
    }
    res = req.get(QUOTES_URL, params=params, timeout=TIMEOUT)
    res.raise_for_status()
    quotes = pd.DataFrame(res.json()).set_index(["src", "dst"])
    return quotes


USER_STATE_FLOAT_COLS = [
    "collateral",
    "depositedCollateral",
    "collateralUp",
    "loss",
    "lossPct",
    "debt",
    "health",
]
USER_STATE_INT_COLS = ["n", "n1", "n2", "timestamp"]


### HISTORICAL ANALYSIS FOR DEBT & LIQUIDITY SAMPLING ###

get_user_snapshots = sync(get_user_snapshots)
stableswap_snapshot = sync(_stableswap_snapshot)


def get_historical_user_snapshots(address: str, start: int, end: int) -> pd.DataFrame:
    """
    Get user state snapshots for a given address.
    """
    loop = get_event_loop()
    ts = start
    snapshots = []
    while ts <= end:
        snapshots.extend(get_user_snapshots(address, end_ts=ts, event_loop=loop))
        ts += 60 * 60 * 24

    df = pd.DataFrame(snapshots)
    df["user"] = df["user"].apply(lambda x: x["id"])

    for col in USER_STATE_FLOAT_COLS:
        df[col] = df[col].astype(float)

    for col in USER_STATE_INT_COLS:
        df[col] = df[col].astype(int)

    # Don't include negligible positions
    df = df[df["collateral"] > 0]
    df = df[df["debt"] > 1]  # some positions have dust
    df = df[df["health"] > 0]  # filter out unhealthy positions

    df["collateral_log"] = np.log(df["collateral"])
    df["debt_log"] = np.log(df["debt"])
    df = df.dropna(axis=1)

    return df.set_index("id")


def group_user_states(user_states: pd.DataFrame) -> pd.DataFrame:
    """
    Group user states df by datetime.
    """
    user_states["datetime"] = pd.to_datetime(user_states["timestamp"], unit="s")

    return user_states.groupby("datetime").agg(
        debt=("debt", "sum"),
        collateral=("collateral", "sum"),
        num_loans=("debt", "count"),
    )


def extract_pool_stats(_snapshot: dict) -> tuple:
    """
    Get the total supply from a snapshot.
    """
    snapshot = _snapshot.copy()
    snapshot["coins"] = [StableCoin(**coin_kwargs) for coin_kwargs in snapshot["coins"]]
    spool_metadata = CurveStableSwapPoolMetaData(
        snapshot, CurveStableSwapPool, SimCurveStableSwapPool
    )

    spool = SimCurveStableSwapPool(**spool_metadata.init_kwargs())

    return (
        *[b * r / 1e36 for b, r in zip(spool.balances, spool.rates)],
        spool.totalSupply / 1e18,
    )


def make_df_stableswap_stats_df(stats: list) -> pd.DataFrame:
    """
    Convert a list of stats into a DataFrame.
    """
    df = pd.DataFrame(stats, columns=["timestamp", "peg", "crvUSD", "supply"])
    return df.set_index(pd.to_datetime(df["timestamp"], unit="s"))


def get_historical_stableswap_stats(
    addresses: List[str], start: int, end: int
) -> Dict[str, pd.DataFrame]:
    """
    Get the historical balance and total LP token supply
    for input stableswap pools.

    TODO should this really be a list or dict?
    TODO might need to add event loop!
    """
    dfs = defaultdict(list)
    ts = start
    while ts < end:
        snapshots = stableswap_snapshot(addresses, end_ts=ts)
        for i, address in enumerate(addresses):
            dfs[address].append([ts, *extract_pool_stats(snapshots[i])])
        ts += 60 * 60 * 24

    return {address: make_df_stableswap_stats_df(df) for address, df in dfs.items()}


def get_event_loop() -> asyncio.AbstractEventLoop:
    """
    Ensure an event loop is running.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop
