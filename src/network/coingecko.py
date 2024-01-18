"""Provides functions for fetching Coingecko API data."""

from datetime import datetime
from typing import List
import requests as req
from curvesim.network.coingecko import coin_ids_from_addresses_sync
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from ..logging import get_logger

COINGECKO_URL = "https://api.coingecko.com/api/v3/"

logger = get_logger(__name__)


# A hack to minimize API calls
KNOWN_IDS_MAP = {
    "usd-coin": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "tether": "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "paxos-standard": "0x8e870d67f660d95d5be530380d0ec0bd388289e1",
    "true-usd": "0x0000000000085d4780b73119b644ae5ecd22b376",
    "weth": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "wrapped-bitcoin": "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599",
    "wrapped-steth": "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0",
    "staked-frax-ether": "0xac3e018457b222d93114458476f3e3416abbe38f",
    "tbtc": "0x18084fba666a33d37592fa2633fd49a74dd93a88",
}


MAX_RETRIES = 8
TIMEOUT = 30


@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1.5, min=2, max=60),
    retry=retry_if_exception_type(req.HTTPError),
)
def _get(url: str, params: dict) -> dict:
    """GET data from coingecko API."""
    res = req.get(url, params=params, timeout=TIMEOUT)
    res.raise_for_status()  # retry if rate limit error
    return res.json()


def get_current_prices(coin_ids: List[str]) -> dict:
    """
    Get current price data from Coingecko API.

    Parameters
    ----------
    coin_ids : List[str]
        Coin ids for coingecko API.

    Returns
    -------
    dict
        List of price data.
    """
    url = COINGECKO_URL + "simple/price"
    coin_ids = [
        coin_ids_from_addresses_sync(coin, "mainnet") if "0x" in coin else coin
        for coin in coin_ids
    ]
    params = {
        "ids": ",".join(coin_ids),
        "vs_currencies": ",".join(["usd"] * len(coin_ids)),
    }
    return {k: v["usd"] for k, v in _get(url, params).items()}


def get_historical_prices(coin_id: str, start: int, end: int) -> List:
    """
    Get historical price data from Coingecko API.

    Parameters
    ----------
    coin_id : str
        Coin id for coingecko API.
    start : int
        Unix timestamp in milliseconds.
    end : int
        Unix timestamp in milliseconds.

    Returns
    -------
    list
        List of price data.

    Note
    ----
    The coingecko API returns data in the following
    granularity:
        1 day from current time = 5-minutely data
        1 day from anytime (except from current time) = hourly data
        2-90 days from current time or anytime = hourly data
        above 90 days from current time or anytime = daily data (00:00 UTC)
    """
    url = COINGECKO_URL + f"coins/{coin_id}/market_chart/range"
    params = {"vs_currency": "usd", "from": start, "to": end}
    return _get(url, params)["prices"]


def get_prices_df(
    coins: List[str], start: int, end: int, freq: str = "1d"
) -> pd.DataFrame:
    """
    Get price data from coingecko API and convert
    into a formatted DataFrame.

    Parameters
    ----------
    coin_id : List[str]
        Coin id from coingecko API or Ethereum address.
    start : int
        Unix timestamp in milliseconds.
    end : int
        Unix timestamp in milliseconds.
    freq : str
        Frequency of price data. Default is daily.

    Returns
    -------
    df : pd.DataFrame
        DataFrame of price data.
    """
    dfs = []
    for i, coin in enumerate(coins):
        if "0x" in coin:
            # Convert Ethereum address to Coingecko coin id
            coin = coin_ids_from_addresses_sync([coin], "mainnet")[0]
        logger.info(
            "Fetching Coingecko price data for %s...%d/%d", coin, i + 1, len(coins)
        )
        prices = get_historical_prices(coin, start, end)
        cdf = pd.DataFrame(prices, columns=["timestamp", coin])
        cdf.index = pd.Index(pd.to_datetime(cdf["timestamp"], unit="ms"))
        cdf.index.name = "datetime"
        cdf.drop(["timestamp"], axis=1, inplace=True)
        cdf = cdf.resample(freq).mean()
        dfs.append(cdf)
    df = pd.concat(dfs, axis=1)
    df["timestamp"] = df.index
    df["timestamp"] = df["timestamp"].apply(lambda x: int(datetime.timestamp(x)))
    df = df.ffill()
    return df


def address_from_coin_id(coin_id: str, chain: str = "ethereum") -> str:
    """Map Coingecko coin_id to an address"""

    # Check if we already know the address
    if chain == "ethereum" and coin_id in KNOWN_IDS_MAP:
        return KNOWN_IDS_MAP[coin_id]

    logger.warning(
        "Fetching %s address from Coingecko API. Add to KNOWN_IDS_MAP", coin_id
    )
    url = COINGECKO_URL + f"coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "false",
        "community_data": "false",
        "developer_data": "false",
        "sparkline": "false",
    }
    res = _get(url, params)["platforms"]
    return res[chain]
