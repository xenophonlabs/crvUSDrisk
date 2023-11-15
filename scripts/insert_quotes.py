import os

# import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from src.db.datahandler import DataHandler
from src.network.oneinchquotes import OneInchQuotes
from src.configs.config import ALL

load_dotenv()
INCH_API_KEY = os.getenv("INCH_API_KEY")

# TODO implement logging
# logging.basicConfig(filename="./logs/quotes.log", level=logging.INFO, format='%(asctime)s %(message)s')

# Temp for testing
# dh = DataHandler()
# df = pd.read_csv("./data/1inch/quotes.csv")
# dh.insert_quotes(df)

# Set up a cron job that runs every hour
# TODO make more granular, need lower API rate limits
# currently 1RPS means it takes ~30 min to query 72 pairs 20 times each


def main():
    dt = int(datetime.now().timestamp())
    quoter = OneInchQuotes(INCH_API_KEY, ALL, calls=20)
    payload = quoter.all_quotes(list(ALL.keys()))
    # TODO temporarily caching into a csv
    # remove this
    df = quoter.to_df(payload, fn=f"./data/1inch/{dt}.csv")
    dh = DataHandler()
    print("Inserting...")
    dh.insert_quotes(df)


if __name__ == "__main__":
    main()
