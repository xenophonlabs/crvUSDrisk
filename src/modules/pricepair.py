class PricePair:
    """
    @dev frankly this doesn't seem necessary but let's keep
    it for now and delete later if we can.
    """

    __slots__ = (
        "pool",  # CurvePool object
        "is_inverse",
    )

    def __init__(self, pool):
        self.pool = pool
        if pool.metadata["coins"]["names"][1] == "crvUSD":
            self.is_inverse = False
        else:
            self.is_inverse = True

    def get_p(self):
        """
        @return stablecoins/crvUSD price from CurvePool
        """
        if self.is_inverse:
            return self.pool.dydx(0, 1)
        else:
            return self.pool.dydx(1, 0)

    def price_oracle(self):
        """
        @notice return EMA price of pool.
        @dev TODO Need to add EMA functionality to curvesim
        StableSwap Pools. For now using spot.
        """
        return self.get_p()
