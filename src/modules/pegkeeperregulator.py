from typing import List
from .aggregator import AggregateStablePrice
from .pegkeeper import PegKeeper
from .pricepair import PricePair
from curvesim.pool.stableswap import CurvePool
import math

CRVUSD_ADDRESS = '0xf939E0A03FB07F59A73314E73794Be0E57ac1b4E' 

class PegKeeperRegulator:

    __slots__ = (
        # === Dependencies === #
        'aggregator', # aggregator object

        # === State Variables === #
        'price_deviation', # max price deviation to mint/burn
        'price_pairs', # list of CurvePool objects
    )

    def __init__(
            self,
            aggregator: AggregateStablePrice,
            deviation: float=5*10**(18-4), # 0.0005 = 0.05%
            pools: List[CurvePool]=[],
        ) -> None:

        self.aggregator = aggregator
        self.price_deviation = deviation
        self.price_pairs = [PricePair(pool) for pool in pools]

    def get_price(self, pair: PricePair):
        """
        @return stablecoins/crvUSD price from CurvePool
        """
        return pair.get_p()
    
    def get_price_oracle(self, pair):
        """
        @return EMA price
        """
        return pair.price_oracle()

    def price_in_range(self, p0: float, p1: float):
        """
        @notice checks that EMA price is within range from spot price.
        @param p0 EMA price (oracle) or spot price
        @param p1 EMA price (oracle) or spot price
        @return True if price is in range, False otherwise
        """
        return abs(p0 - p1) <= self.price_deviation
    
    def provide_allowed(self, pk: PegKeeper):
        """
        @notice Allow Peg Keeper to provide stablecoin to the pool.
        @param pk Peg Keeper object
        @return True if provide is allowed, False otherwise
        @dev Checks that: 
            1) current price in range of oracle in case of spam-attack
            2) current price location among other pools in case of contrary coin depeg
            3) stablecoin price is above 1
        """
        if self.aggregator.price() < 1e18: 
            return False
        
        price = math.inf
        largest_price = 0
        for pair in self.price_pairs:
            pair_price = self.get_price_oracle(pair)
            if pair.pool.address == pk.pool_address:
                price = pair_price
                if self.price_in_range(pair_price, self.get_price(pair)):
                    return False
                continue
            largest_price = max(largest_price, pair_price)

        return largest_price >= (price - 3 * 10 ** (18 - 4))

    def withdraw_allowed(self, pk: PegKeeper):
        """
        @notice Allow Peg Keeper to withdraw stablecoin from the pool
        @param pk Peg Keeper object
        @return True if withdraw is allowed, False otherwise
        @dev Checks
            1) current price in range of oracle in case of spam-attack
            2) stablecoin price is below 1
        """
        if self.aggregator.price() > 1e18: 
            return False

        for pair in self.price_pairs:
            if pair.pool.address == pk.pool.address:
                return self.price_in_range(self.get_price(pair), self.get_price_oracle(pair))
        return False  # dev: not found
    