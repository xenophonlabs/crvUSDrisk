"""
Utility functions.
"""


def get_crvUSD_index(pool):
    """
    Return index of crvUSD in pool.
    """
    symbols = [c.symbol for c in pool.coins]
    return symbols.index("crvUSD")
