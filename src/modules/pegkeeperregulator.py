class PegKeeperRegulator:

    def __init__(self):
        pass

    def get_price(self, pair):
        # implements stableswap get_price
        # dx_0 / dx_1 only, however can have any number of coins in pool
        if self.pair_prices.get(pair)!=None:
            return self.pair_prices[pair]
        else: return False
    
    def get_price_oracle(self,pair):
        if self.pair_oracle_prices.get(pair)!=None:
            return self.pair_oracle_prices[pair]
        else: return False

    def price_in_range(self,p0,p1):
        # NOTE: we think p0 is p and p1 is p_oracle
         # |p1 - p0| <= deviation
        # -deviation <= p1 - p0 <= deviation
        # 0 < deviation + p1 - p0 <= 2 * deviation
        # NOTE: they mightve swapped the order of p0 and p1 here
        return abs(p1 - p0) <= self.price_deviation
    
    def provide_allowed(self,pk):
        # Checks that: 
        # 1) current price in range of oracle in case of spam-attack
        # 2) current price location among other pools in case of contrary coin depeg
        # 3) stablecoin price is above 1

        if self.aggregator.price() < 1e18: return False
        price = sys.maxsize
        largest_price = 0
        # iterate through all pairs for  smallest_price: uint256 = max_value(uint256)
        # return whether the price is greater than the smallest price for all pairs
        # @TODO: simplify this logic
        for pair in self.price_pairs:
            pair_price = self.get_price_oracle(pair)
            pool_match = (self.pool_addresses[pair] == pk)
            
            if pool_match and not self.price_in_range(price, self.get_price(pair)): 
                return False
            if pool_match and self.price_in_range(price, self.get_price(pair)): 
                continue               
            if not pool_match and largest_price < pair_price: 
                largest_price = pair_price
            if not pool_match and largest_price >= pair_price:
                pass
        return largest_price >= (price - 3 * 10 ** (18 - 4))

    def withdraw_allowed(self,pk):
        """
        @notice Allow Peg Keeper to withdraw stablecoin from the pool
        Checks
            1) current price in range of oracle in case of spam-attack
            2) stablecoin price is below 1
        """
        if self.aggregator.price() > 1e18: return False

        for pair in self.price_pairs:
            pool_match = (self.pool_addresses[pair] == pk)
            if pool_match:
                return self.price_in_range(self.get_price(pair), self.get_price_oracle(pair))
        return False  # dev: not found