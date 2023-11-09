# Start with high level prices
# TODO drill down to higher granularity price fluctuations
# Example: what happens when we get minutely price data?
# Think: PK has several short-timeframe limitations!

from datetime import datetime
import requests as req
import pandas as pd
from ..exceptions import coingeckoRateLimitException
from curvesim.network.coingecko import coin_ids_from_addresses_sync

URL = "https://api.coingecko.com/api/v3/"


def get_price(coin_id: str, start: int, end: int) -> list:
    """
    Get price data from coingecko API.

    Parameters
    ----------
    coin_id : str
        Coin id from coingecko API.
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
    url = URL + f"coins/{coin_id}/market_chart/range"
    p = {"vs_currency": "usd", "from": start, "to": end}
    r = req.get(url, params=p)
    if r.status_code == 200:
        return r.json()["prices"]
    elif r.status_code == 429:
        raise coingeckoRateLimitException("Coingecko API Rate Limit Exceeded.")
    else:
        raise RuntimeError(f"Request failed with status code {r.status_code}.")


def get_prices_df(coins: str, start: int, end: int, freq: str = "1d") -> pd.DataFrame:
    """
    Get price data from coingecko API and convert
    into a formatted DataFrame.

    Parameters
    ----------
    coin_id : str
        Coin id from coingecko API or Ethereum address.
    start : int
        Unix timestamp in milliseconds.
    end : int
        Unix timestamp in milliseconds.
    freq : Optional[str]
        Frequency of price data. Default is daily.

    Returns
    -------
    df : pd.DataFrame
        DataFrame of price data.
    """
    dfs = []
    for coin in coins:
        if "0x" in coin:
            # Convert Ethereum address to Coingecko coin id
            coin = coin_ids_from_addresses_sync([coin], "mainnet")[0]
        prices = get_price(coin, start, end)
        cdf = pd.DataFrame(prices, columns=["timestamp", coin])
        cdf.index = pd.to_datetime(cdf["timestamp"], unit="ms")
        cdf.index.name = "datetime"
        cdf.drop(["timestamp"], axis=1, inplace=True)
        cdf = cdf.resample(freq).mean()
        dfs.append(cdf)
    df = pd.concat(dfs, axis=1)
    df["timestamp"] = df.index
    df["timestamp"] = df["timestamp"].apply(lambda x: int(datetime.timestamp(x)))
    df = df.fillna(method="ffill")
    return df
