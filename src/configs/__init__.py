"""
Master config with basic constants.
"""

import os
import json
from .tokens import ADDRESSES, TOKEN_DTOs, STABLE_CG_IDS, CRVUSD_DTO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = [
    "COINGECKO_URL",
    "ADDRESSES",
    "TOKEN_DTOs",
    "ADDRESS_TO_SYMBOL",
    "SYMBOL_TO_ADDRESS",
    "STABLE_CG_IDS",
    "CRVUSD_DTO",
]

COINGECKO_URL = "https://api.coingecko.com/api/v3/"

# Convenience maps
ADDRESS_TO_SYMBOL = {k: v.symbol for k, v in TOKEN_DTOs.items()}
SYMBOL_TO_ADDRESS = {v.symbol: k for k, v in TOKEN_DTOs.items()}

# Constants
DEFAULT_PROFIT_TOLERANCE = 1  # one dollah


def get_scenario_config(scenario: str) -> dict:
    """Return scenario config dict."""
    fn = os.path.join(BASE_DIR, "scenarios", scenario + ".json")
    with open(fn, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def get_price_config(freq: str) -> dict:
    """Return the latest price config dict."""
    dir_ = os.path.join(BASE_DIR, "prices", freq)
    files = [os.path.join(dir_, f) for f in os.listdir(dir_)]
    if not files:
        raise ValueError(f"Price configs not generated for {freq}.")
    fn = sorted(files, reverse=True)[0]
    with open(fn, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config
