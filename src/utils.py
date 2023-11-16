"""
Dump for all uncategorized utility functions
"""


def slippage(size, sigma):
    # TODO understand what the actual trades would be
    # for the arbitrage. Would they
    # FIXME Ignoring volatility for now
    m = 1.081593506690093e-06
    b = 0.0004379110082802476
    return m * size + b


def get_crvUSD_index(pool):
    """
    Return index of crvUSD in pool.
    """
    return pool.metadata["coins"]["names"].index("crvUSD")


def external_swap(x, y, swap, fee, y_in):
    """
    @notice account for slippage when trading against open market
    Assuming that we trade against Uniswap is a conservative assumption
    since not accounting for CEX.
    However, we are not accounting for crvUSD liquidity in Curve pools yet.
    # Fit this curve
    # f(x | sigma) = c * sigma * x <- Gauntlet found this empirically in Compound report
    # f(x | sigma) = c * sigma * x**0.5 <- TradFi empirical finding
    # f(x | sigma) = c * sigma * x**2 <- Perhaps more like a simple CFMM
    """
    # TODO: Need to incorporate concentrated liquidity (e.g., no more liquidity beyond ]a, b[)
    # TODO: Need to incorproate the StableSwap invariant for crvUSD pool liquidity
    k = x * y

    if y_in:
        new_y = y + swap
        new_x = k / new_y
        out = (x - new_x) * (1 - fee)
    else:
        new_x = x + swap
        new_y = k / new_x
        out = (y - new_y) * (1 - fee)

    return out
