from dotenv import load_dotenv
import argparse
import os
from src.utils.ccxtdatafetcher import CCXTDataFetcher
from datetime import datetime

load_dotenv()
coinbase_pro_api_key = os.getenv("COINBASE_PRO_API_KEY")
coinbase_pro_api_secret = os.getenv("COINBASE_PRO_API_SECRET")
coinbase_pro_api_password = os.getenv("COINBASE_PRO_API_PWD")


def main(exchange, symbol, since, end):
    datafetcher = CCXTDataFetcher(
        coinbase_pro_api_key, coinbase_pro_api_secret, coinbase_pro_api_password
    )

    if exchange == "coinbasepro":
        df = datafetcher.fetch_trades_coinbasepro(symbol, since)
    elif exchange == "binance":
        df = datafetcher.fetch_trades_binance(symbol, since, end=end)
    else:
        raise ValueError("Exchange not supported.")

    fn = f"./data/trades/{exchange}_{symbol.replace('/', '_')}_{int(datetime.fromisoformat(since).timestamp())}.gzip"
    df.to_parquet(fn, compression="gzip")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch CEX trades using CCXT.")
    parser.add_argument("exchange", type=str, help="coinbasepro or binance.")
    parser.add_argument("symbol", type=str, help="Symbol like ETH/USDC.")
    parser.add_argument("since", type=str, help="Start date YYYY-MM-DD HH:MM:SS.")
    parser.add_argument(
        "-e", "--end", type=str, help="End date YYYY-MM-DD HH:MM:SS.", required=False
    )
    args = parser.parse_args()
    main(args.exchange, args.symbol, args.since, args.end)
