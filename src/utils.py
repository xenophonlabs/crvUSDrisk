"""
Dump for all uncategorized utility functions
"""
from .configs.config import ALL, ADDRESS_TO_SYMBOL

def get_decimals_from_config(address):
    """
    Get the decimals for a token from the config.
    """
    return ALL[ADDRESS_TO_SYMBOL[address]]["decimals"]


def get_crvUSD_index(pool):
    """
    Return index of crvUSD in pool.
    """
    return pool.coin_names.index("crvUSD")
