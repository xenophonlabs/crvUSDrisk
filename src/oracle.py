import math

class Oracle:
    """
    @notice simple oracle for llamma
    TODO incorporate chainlink limits
    TODO incorporate StableSwap and TriCrypto prices
    TODO incorporate TriCrypto pool liquidity <- e.g., a USDC depeg would affect ETH/BTC/USDC liquidity
    """

    __slots__ = (
        'last_timestamp', # timestamp since last update
        # 'last_tvl', # tvl for last update
        # 'TVL_MA_TIME', # EMA window for tvl
        'last_price', # temp price for last update to do bootleg EMA
        'PRICE_MA_TIME', # temp EMA window for price
        'v', # verbose
    )

    def __init__(
            self,
            PRICE_MA_TIME: int,
            v: bool = False
        ):
            self.PRICE_MA_TIME = PRICE_MA_TIME
            self.v = v

    def price(self):
        return self.last_price

    def update(
            self, 
            t: int, 
            price: float
        ) -> float:
        """
        @notice for now, we are feeding in an ETH/USD price 
        from a GBM and applying EMA.
        @param t current timestamp
        @param price current spot price
        @return EMA price
        NOTE ideally would find a way to use DataFrame ops for this instead
        TODO the correct implementation will be:
        1. Generate liquidity curves for TriCrypto pools
        2. Generate liquidity curves for PK pools
        3. Compute EMA TVL for TriCrypto pools
        4. Compute EMA prices for TriCrypto pools
        5. Compute EMA prices for PK pools
        6. Transform TriCrypto and PK prices into liquidity-weighted ETH/crvUSD
        7. Multiply by StableSwap agg prices to get ETH/USD
        """
        if not hasattr(self, 'last_timestamp') or not hasattr(self, 'last_price'):
            # No EMA to perform
            self.last_timestamp = t
            self.last_price = price
            return price

        if self.last_timestamp < t:
            alpha = math.exp(-(t - self.last_timestamp) / self.PRICE_MA_TIME)
            price_EMA = alpha * self.last_price + (1 - alpha) * price
            # Updates
            self.last_timestamp = t
            self.last_price = price_EMA

        return self.last_price