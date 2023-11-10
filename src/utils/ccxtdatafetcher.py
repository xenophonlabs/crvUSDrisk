import ccxt as ccxt
from datetime import datetime
import pandas as pd
from ..exceptions import ccxtInvalidSymbolException
import math


class CCXTDataFetcher:
    """
    Pull trade and OHLCV data from CCXT.
    Supported exchanges:
        Binance
        Coinbasepro (required API Key)
    """

    def __init__(
        self,
        coinbase_pro_api_key,
        coinbase_pro_api_secret,
        coinbase_pro_api_password,
        enable_rate_limit: bool = True,
    ):
        self.coinbasepro = ccxt.coinbasepro(
            {
                "apiKey": coinbase_pro_api_key,
                "secret": coinbase_pro_api_secret,
                "password": coinbase_pro_api_password,
                "enableRateLimit": enable_rate_limit,
            }
        )
        self.coinbasepro.load_markets()

        self.binance = ccxt.binance(
            {"enableRateLimit": enable_rate_limit}
        )  # Can't be in the U.S.
        self.binance.load_markets()

    def since(self, dt: datetime, exchange_id: str = "coinbasepro") -> int:
        """
        Use exchange.parse8601 for given exchange to convert python
        dt to UNIX millisecond timestamps.
        """
        exchange = getattr(self, exchange_id)
        return exchange.parse8601(datetime.isoformat(dt))

    def to_strftime(self, dt) -> str:
        if isinstance(dt, int):
            # Assumed mili
            dt = datetime.fromtimestamp(dt / 1000)
        return dt.strftime("%d/%m/%Y, %H:%M:%S")

    def fetch_trades_coinbasepro(self, symbol, since):
        """
        `fetchTrades` on coinbasepro returns the most
        recent page of trades. You must then walk backwards
        through the pages until your desired `since`.

        Parameters
        ----------
        symbol : str
            token0/token1.
        since : int or datetime or 8601 str
            Earliest trade timestamp in milliseconds.

        Returns
        -------
        pd.DataFrame
            Trades.
        """

        start_time = datetime.now()
        exchange = self.coinbasepro

        if isinstance(since, datetime):
            since = self.since(since)
        if isinstance(since, str):
            since = exchange.parse8601(since)

        print(
            f"\nQuerying {symbol} trades from Coinbase Pro between {self.to_strftime(start_time)} and {self.to_strftime(since)}.\n"
        )

        total = int(start_time.timestamp() * 1000) - since
        bars = [x for x in range(101)]
        print(f"Progress: {bars.pop(0)}%", end="\r")

        param_key = ""
        param_value = ""
        last_trade_ts = math.inf
        trades = []
        if symbol in exchange.markets:
            while since < last_trade_ts:
                new_trades = exchange.fetch_trades(
                    symbol, params={param_key: param_value}
                )
                if not len(new_trades):
                    break
                trades.extend(new_trades)
                after = exchange.last_response_headers.get("Cb-After")
                if after:
                    param_key = "after"
                    param_value = after
                    last_trade_ts = new_trades[0]["timestamp"]
                else:
                    break  # Last page
                progress = round((1 - (last_trade_ts - since) / total) * 100)
                if len(bars) and progress in bars:
                    bars = bars[bars.index(progress) + 1 :]
                    print(f"Progress: {progress}%", end="\r")
        else:
            raise ccxtInvalidSymbolException

        end_time = datetime.now()

        print(f"\nFinished. Time taken: {end_time - start_time}\n")

        return CCXTDataFetcher.trades_to_df(trades)

    def fetch_trades_binance(self, symbol, since, end=None):
        start_time = datetime.now()
        exchange = self.binance

        if isinstance(since, datetime):
            since = self.since(since)
        if isinstance(since, str):
            since = exchange.parse8601(since)
        if isinstance(end, datetime):
            end = self.since(end)
        if isinstance(end, str):
            end = exchange.parse8601(end)

        cur = int(datetime.now().timestamp() * 1000)
        end = min(end, cur) if end else cur

        print(
            f"\nQuerying {symbol} trades from Binance between {self.to_strftime(since)} and {self.to_strftime(end)}.\n"
        )

        total = end - since
        bars = [x for x in range(101)]
        print(f"Progress: {bars.pop(0)}%", end="\r")
        trades = []
        if symbol in exchange.markets:
            while since < end:
                try:
                    new_trades = exchange.fetch_trades(symbol, since=since)
                    if not len(new_trades):
                        break
                    trades.extend(new_trades)
                    since = trades[-1]["timestamp"]  # "latest" trade

                    progress = round((1 - (end - since) / total) * 100)
                    if len(bars) and progress in bars:
                        bars = bars[bars.index(progress) + 1 :]
                        print(f"Progress: {progress}%", end="\r")
                except ccxt.RateLimitExceeded:
                    exchange.sleep(10000)
                except Exception:
                    raise
        else:
            raise ccxtInvalidSymbolException

        end_time = datetime.now()

        print(f"\n\nFinished. Time taken: {end_time - start_time}\n")

        return CCXTDataFetcher.trades_to_df(trades)

    def trades_to_df(trades: list) -> pd.DataFrame:
        df = pd.DataFrame(trades)
        df.drop(["info"], inplace=True)
        df.set_index("datetime", inplace=True)
        df.sort_index(inplace=True)
        return df
