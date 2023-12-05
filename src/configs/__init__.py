"""
Master config with basic constants.
"""

import os
import json
from dotenv import load_dotenv
from .tokens import ADDRESSES, TOKEN_DTOs, STABLE_CG_IDS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = [
    "URI",
    "COINGECKO_URL",
    "ADDRESSES",
    "TOKEN_DTOs",
    "ADDRESS_TO_SYMBOL",
    "SYMBOL_TO_ADDRESS",
    "STABLE_CG_IDS",
]

load_dotenv()

username = os.getenv("PG_USERNAME")
password = os.getenv("PG_PASSWORD")
database = os.getenv("PG_DATABASE")

# TODO convert to remote DB query
URI = (
    f"postgresql://{username}:{password}@localhost/{database}"  # defaults to port 5432
)

COINGECKO_URL = "https://api.coingecko.com/api/v3/"

# Convenience maps
ADDRESS_TO_SYMBOL = {k: v.symbol for k, v in TOKEN_DTOs.items()}
SYMBOL_TO_ADDRESS = {v.symbol: k for k, v in TOKEN_DTOs.items()}

# Constants
DEFAULT_PROFIT_TOLERANCE = 1  # one dollah


def get_config(fn: str, dir_: str) -> dict:
    """Return config dict."""
    fn = os.path.join(BASE_DIR, dir_, fn + ".json")
    with open(fn, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config
