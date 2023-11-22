"""
Dump for all uncategorized utility functions
"""

def get_crvUSD_index(pool):
    """
    Return index of crvUSD in pool.
    """
    return pool.coin_names.index("crvUSD")
