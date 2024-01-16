"""
Master config with basic constants.
"""

import os
import json
import pickle
from datetime import datetime
from scipy.stats import gaussian_kde
from .scenarios import SCENARIOS
from .tokens import ADDRESSES, TOKEN_DTOs, STABLE_CG_IDS, CRVUSD_DTO
from ..network.coingecko import get_current_prices
from ..logging import get_logger

logger = get_logger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

__all__ = [
    "ADDRESSES",
    "TOKEN_DTOs",
    "ADDRESS_TO_SYMBOL",
    "SYMBOL_TO_ADDRESS",
    "STABLE_CG_IDS",
    "CRVUSD_DTO",
]

# Convenience maps
ADDRESS_TO_SYMBOL = {k: v.symbol for k, v in TOKEN_DTOs.items()}
SYMBOL_TO_ADDRESS = {v.symbol: k for k, v in TOKEN_DTOs.items()}

# Constants
DEFAULT_PROFIT_TOLERANCE = 1  # one dollah
# TODO modify tolerance to account for gas (and speed up sim)

# LLAMMAs
LLAMMA_WETH = "0x1681195c176239ac5e72d9aebacf5b2492e0c4ee"
LLAMMA_WBTC = "0xe0438eb3703bf871e31ce639bd351109c88666ea"
LLAMMA_SFRXETH = "0xfa96ad0a9e64261db86950e2da362f5572c5c6fd"
LLAMMA_WSTETH = "0x37417b2238aa52d0dd2d6252d989e728e8f706e4"
LLAMMA_TBTC = "0xf9bd9da2427a50908c4c6d1599d8e62837c2bcb0"

# Aliases
LLAMMA_WETH_ALIAS = "weth"
LLAMMA_WBTC_ALIAS = "wbtc"
LLAMMA_SFRXETH_ALIAS = "sfrxeth"
LLAMMA_WSTETH_ALIAS = "wsteth"
LLAMMA_TBTC_ALIAS = "tbtc"

LLAMMA_ALIASES = {
    LLAMMA_WETH_ALIAS: LLAMMA_WETH,
    LLAMMA_WBTC_ALIAS: LLAMMA_WBTC,
    LLAMMA_SFRXETH_ALIAS: LLAMMA_SFRXETH,
    LLAMMA_WSTETH_ALIAS: LLAMMA_WSTETH,
    LLAMMA_TBTC_ALIAS: LLAMMA_TBTC,
}
ALIASES_LLAMMA = {v: k for k, v in LLAMMA_ALIASES.items()}

MODELLED_MARKETS = ["wbtc", "weth", "sfrxeth", "wsteth"]

CRVUSD_USDC_ADDRESS = "0x4dece678ceceb27446b35c672dc7d61f30bad69e"
CRVUSD_USDT_ADDRESS = "0x390f3595bca2df7d23783dfd126427cceb997bf4"
CRVUSD_TUSD_ADDRESS = "0x34d655069f4cac1547e4c8ca284ffff5ad4a8db0"
CRVUSD_USDP_ADDRESS = "0xca978a0528116dda3cba9acd3e68bc6191ca53d0"

STABLESWAP_ADDRESSES = [
    CRVUSD_USDC_ADDRESS,
    CRVUSD_USDT_ADDRESS,
    CRVUSD_TUSD_ADDRESS,
    CRVUSD_USDP_ADDRESS,
]

CRVUSD_USDC_ADDRESS_ALIAS = "usdc_stableswap"
CRVUSD_USDT_ADDRESS_ALIAS = "usdt_stableswap"
CRVUSD_TUSD_ADDRESS_ALIAS = "tusd_stableswap"
CRVUSD_USDP_ADDRESS_ALIAS = "usdp_stableswap"

STABLESWAP_ALIASES = {
    CRVUSD_USDC_ADDRESS: CRVUSD_USDC_ADDRESS_ALIAS,
    CRVUSD_USDT_ADDRESS: CRVUSD_USDT_ADDRESS_ALIAS,
    CRVUSD_TUSD_ADDRESS: CRVUSD_TUSD_ADDRESS_ALIAS,
    CRVUSD_USDP_ADDRESS: CRVUSD_USDP_ADDRESS_ALIAS,
}
ALIASES_STABLESWAP = {v: k for k, v in STABLESWAP_ALIASES.items()}


def get_scenario_config(scenario: str) -> dict:
    """Return scenario config dict."""
    return SCENARIOS[scenario.lower()]


def get_borrower_kde(market: str, start: int, end: int) -> gaussian_kde:
    """
    Get the KDE object for sampling borrowers.
    """
    if "0x" in market:
        market = ALIASES_LLAMMA[market]
    fn = os.path.join(BASE_DIR, "borrowers", market.lower(), f"{start}_{end}.pkl")
    with open(fn, "rb") as f:
        kde = pickle.load(f)
    return kde


def get_liquidity_config(start: int, end: int) -> dict:
    """
    Return the matching liquidity config dict.
    """
    fn = os.path.join(BASE_DIR, "liquidity", f"{start}_{end}.json")
    with open(fn, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg


def get_price_config(freq: str, start: int, end: int) -> dict:
    """
    Return the latest price config dict.
    """
    dir_ = os.path.join(BASE_DIR, "prices", freq)
    files = [os.path.join(dir_, f) for f in os.listdir(dir_)]

    if not files:
        raise ValueError(f"No price configs generated for {freq}.")

    target = f"{start}_{end}.json"
    fn = None
    found = False
    for fn_ in files:
        if fn_.split("/")[-1] == target:
            fn = fn_
            found = True
    if fn is None:
        fn = sorted(files, reverse=True)[0]

    with open(fn, "r", encoding="utf-8") as f:
        config = json.load(f)

    coin_ids = list(config["params"].keys())
    config["curr_prices"] = get_current_prices(coin_ids)

    if not found:
        logger.warning(
            "No price config found from %s to %s. Using config from %s to %s instead.\
            Please run `gen_price_config` to generate the requested config.",
            datetime.fromtimestamp(start),
            datetime.fromtimestamp(end),
            datetime.fromtimestamp(config["start"]),
            datetime.fromtimestamp(config["end"]),
        )

    return config
