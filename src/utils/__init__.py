"""
Utility functions.
"""


def get_crvusd_index(pool):
    """
    Return index of crvusd in pool.
    """
    symbols = [c.symbol for c in pool.coins]
    return symbols.index("crvUSD")
