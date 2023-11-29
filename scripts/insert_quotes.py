import os

import logging
from datetime import datetime
from dotenv import load_dotenv

from src.db.datahandler import DataHandler
from src.network.oneinch import OneInchQuotes
from src.configs import TOKEN_DTOs

load_dotenv()
INCH_API_KEY = os.getenv("1INCH_API_KEY")
assert INCH_API_KEY, "Missing API Key in .env"

# TODO implement logging
logging.basicConfig(
    filename="./logs/quotes.log", level=logging.INFO, format="%(asctime)s %(message)s"
)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("urllib3").propagate = False

# TODO make more granular, need lower API rate limits
# currently 1RPS means it takes ~30 min to query 72 pairs 20 times each

# TODO start querying other token pairs for markets
# we might want to include in the future.


def main():
    dt = int(datetime.now().timestamp())
    print(
        f"Fetching quotes on {datetime.fromtimestamp(dt).strftime('%m/%d/%Y, %H:%M:%S')} UTC"
    )
    quoter = OneInchQuotes(INCH_API_KEY, TOKEN_DTOs, calls=20)
    payload = quoter.all_quotes(list(TOKEN_DTOs.keys()))
    df = quoter.to_df(payload)
    dh = DataHandler()
    print("Inserting...")
    dh.insert_quotes(df)


if __name__ == "__main__":
    main()
