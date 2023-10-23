import numpy as np
import time
import math
from .pricepair import PricePair
from typing import List
from curvesim.pool import CurvePool

PRECISION = 1e18

class AggregateStablePrice:

    __slots__ = (
        # === Dependencies === #
        'price_pairs',

        # === Parameters === #
        'sigma', # sensitivity parameter for price aggregation
        'min_liquidity', # minimum liquidity to consider pool for aggregation
        'tvl_ma_time', # time to calculate moving average of TVL

        # === State Variables === #
        'last_price', 
        'last_timestamp',
        'last_tvl'
    )

    def __init__(
            self,
            pools: List[CurvePool],
            sigma: float,
            tvl_ma_time: float=50000,
            min_liquidity: float=100,
        ):
        self.price_pairs = [PricePair(pool) for pool in pools]
        self.sigma = sigma
        self.tvl_ma_time = tvl_ma_time
        self.min_liquidity = min_liquidity
        # TODO need to incorporate eixsting last_price, timestamp, tvl at initialization

    def ema_tvl(self, ts: int=None) -> List[float]:
        """
        Calculate the exponential moving average (EMA) of the 
        total value locked (TVL) of each pool.
        
        Parameters
        ----------
        ts : int, optional
            The timestamp to calculate the EMA at. If not provided, 
            the current timestamp will be used.
        
        Returns
        -------
        List[float]
            The EMA of the TVL for each pool.
        """
        ts = ts or int(time.time())

        if not hasattr(self, 'last_timestamp'):
            # No EMA to perform
            self.last_timestamp = ts
            self.last_tvl = [pair.pool.tokens / PRECISION for pair in self.price_pairs]
            return self.last_tvl

        alpha = math.exp(-(ts - self.last_timestamp) / self.tvl_ma_time)
        for i, pair in enumerate(self.price_pairs):
            self.last_tvl[i] = alpha * self.last_tvl[i] + (1 - alpha) * pair.pool.tokens / PRECISION
        
        return self.last_tvl

    def price(self) -> float:
        """
        Return aggregated stablecoin price from Curve pools.
        Methodology:
            1. Compute liquidity weighted average price 
            (using EMA prices from pools).
            2. Compute each pool's deviation from the average price.
            3. Compute weights for each pool based on an exponential
            squared error.

        Returns
        -------
        float
            The aggregated stablecoin price.
        """
        n = len(self.price_pairs)
        prices = D = errors = np.zeros(n) 

        # Compute LWA
        Dsum = DPsum = 0
        for i, pair in enumerate(self.price_pairs):
            if pair.pool.tokens > self.min_liquidity:
                prices[i] = pair.price_oracle() / PRECISION
                D = pair.pool.tokens / PRECISION
                Dsum += D
                DPsum += D * prices[i]
        p_avg = DPsum / Dsum

        # Compute error weights
        for i in range(n):
            errors[i] = ((max(prices[i], p_avg) - min(prices[i], p_avg)) / self.sigma)**2
        min_error = min(errors)

        # Compute error weighted average
        wsum = wpsum = 0
        for i in range(n):
            w = D[i] * math.exp(-(errors[i] - min_error))
            wsum += w
            wpsum += w * prices[i]

        return wpsum / wsum
