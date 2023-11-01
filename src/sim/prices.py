# Start with high level prices
# TODO drill down to higher granularity price fluctuations
# Example: what happens when we get minutely price data?
# Think: PK has several short-timeframe limitations!

from datetime import datetime
import requests as req
import pandas as pd

URL = "https://api.coingecko.com/api/v3/"


def get_price(coin_id, start, end):
    url = URL + f"coins/{coin_id}/market_chart/range"
    p = {"vs_currency": "usd", "from": start, "to": end}
    r = req.get(url, params=p)
    if r.status_code == 200:
        return r.json()["prices"]
    else:
        raise RuntimeError(f"Request failed with status code {r.status_code}.")


def get_prices_df(coins, start, end):
    dfs = []
    for coin in coins:
        prices = get_price(coin, start, end)
        cdf = pd.DataFrame(prices, columns=["timestamp", coin])
        cdf.index = pd.to_datetime(cdf["timestamp"], unit="ms")
        cdf.index.name = "datetime"
        cdf.drop(["timestamp"], axis=1, inplace=True)
        cdf = cdf.resample("1h").mean()
        dfs.append(cdf)
    df = pd.concat(dfs, axis=1)
    df["timestamp"] = df.index
    df["timestamp"] = df["timestamp"].apply(lambda x: int(datetime.timestamp(x)))
    return df
