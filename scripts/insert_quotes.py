import os

from datetime import datetime
from dotenv import load_dotenv
from src.logging import get_logger
from src.db.datahandler import DataHandler
from src.network.oneinch import OneInchQuotes
from src.configs import TOKEN_DTOs

load_dotenv()
INCH_API_KEY = os.getenv("1INCH_API_KEY")
assert INCH_API_KEY, "Missing API Key in .env"

logger = get_logger(__name__)

# TODO make more granular, need lower API rate limits
# currently 1RPS means it takes ~30 min to query 72 pairs 20 times each

# TODO start querying other token pairs for markets
# we might want to include in the future.


def main():
    dt = int(datetime.now().timestamp())
    logger.info(
        f"Fetching quotes on {datetime.fromtimestamp(dt).strftime('%m/%d/%Y, %H:%M:%S')} UTC"
    )
    quoter = OneInchQuotes(INCH_API_KEY, TOKEN_DTOs, calls=20)
    payload = quoter.all_quotes(list(TOKEN_DTOs.keys()))
    df = quoter.to_df(payload)
    dh = DataHandler()
    logger.info("Inserting...")
    dh.insert_quotes(df)


if __name__ == "__main__":
    main()
