"""
Master config with basic constants.
"""

import os
from dotenv import load_dotenv
from .tokens import ADDRESSES, TOKEN_DTOs, STABLE_CG_IDS

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

# Constatns
DEFAULT_PROFIT_TOLERANCE = 1  # one dollah
