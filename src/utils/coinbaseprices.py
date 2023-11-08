import http.client
import json
import pandas as pd
import time
import os


# Fetch ETH Price Data from coinbase
class CoinbasePrices:
    def __init__(self):
        pass

    # initial_time (1483246800): Sunday, January 1, 2017 12:00:00 AM EST
    # initial_time (1641013200): Sunday, January 1, 2022 12:00:00 AM EST
    # gran options (in seconds): 60, 300, 900, 3600, 21600, 86400
    # asset_pair options: ETH-USD, BTC-USD, ETH-BTC, USDT-USDC
    def fetch_prices(
        self,
        asset_pair="USDT-USDC",
        gran=3600,
        initial_time=1641013200,
        path_to_data_dir="../../data/",
    ):
        conn = http.client.HTTPSConnection("api.exchange.coinbase.com")
        payload = ""
        headers = {"User-Agent": "YourAppName/1.0", "Content-Type": "application/json"}
        # rate limit: Requests per second per IP: 10
        delay = 0.105
        current_timestamp = int(time.time())
        projected_end = gran * (int(current_timestamp / gran) + 1)
        filename = (
            path_to_data_dir
            + asset_pair
            + "_price_"
            + str(gran)
            + "_"
            + str(initial_time)
            + "_to_"
            + str(projected_end)
            + ".csv"
        )
        base_df = pd.DataFrame(
            columns=["unix_timestamp", "low", "high", "open", "close", "volume"]
        )
        for start in range(initial_time, current_timestamp, gran):
            try:
                start_time = time.time()
                end = start + gran
                conn.request(
                    "GET",
                    "/products/"
                    + asset_pair
                    + "/candles?granularity="
                    + str(gran)
                    + "&start="
                    + str(start)
                    + "&end="
                    + str(end),
                    payload,
                    headers,
                )
                res = conn.getresponse()
                data = json.loads(res.read().decode("utf-8"))

                df = pd.DataFrame(data)
                if len(df) > 0:
                    df.columns = [
                        "unix_timestamp",
                        "low",
                        "high",
                        "open",
                        "close",
                        "volume",
                    ]
                    df["datetime"] = pd.to_datetime(df["unix_timestamp"], unit="s")

                    base_df = pd.concat([base_df, df], ignore_index=True)
                    base_df = base_df.drop_duplicates()
                    base_df.to_csv(filename)
                else:
                    time.sleep(delay)
                    print("empty_df")

                end_time = time.time()
                elapsed_time = end_time - start_time
                if elapsed_time < delay:
                    time.sleep(delay)

                if start % 100 * gran == 0:
                    os.system("clear")
                    time_perc = round(
                        100 * (start - initial_time) / (projected_end - initial_time), 2
                    )
                    remaining_time_estimate = round(
                        ((projected_end - end) / gran) * max(elapsed_time, delay), 2
                    )
                    print(
                        start,
                        "/",
                        projected_end,
                        str(time_perc) + "%",
                        round(elapsed_time, 2),
                        "seconds",
                        remaining_time_estimate,
                        "seconds remaining",
                    )
            except Exception as e:
                time.sleep(delay)
                print(e)
                print(res.status, res.read().decode("utf-8"))

        base_df.to_csv(filename)
        return base_df
