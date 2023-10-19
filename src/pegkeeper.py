import sys 

class PegKeeper:

    def __init__(self):
        pass
        self.price_deviation = 5 * 10 ** (18 - 4) # 0.0005 = 0.05%
        self.price_pairs = [] # from the vyper code
        self.pair_prices = {} # we created this
        self.pair_oracle_prices = {} # we created this
        self.pool_addresses = {} 
        self.max_price_pairs = 8

        # Time between providing/withdrawing coins
        self.ACTION_DELAY = 15 * 60
        self.ADMIN_ACTIONS_DELAY = 3 * 86400
        
        # NOTE: assume coins[1] is the stablecoin and coins[1].decimals() == 18
        self.PEG_MUL = 10 ** 0
        self.PRECISION = 10 ** 18
        # Calculation error for profit
        self.PROFIT_THRESHOLD = 10 ** 18

        # @TODO: need to implement min and max burn amounts  
        self.min_mint_amount=None
        self.max_burn_amount=None

        # @QUESTION: should we implement a caller_share?
        # self.caller_share = 0  
    
    # @TODO: need to implement stableswap pool add_liquidity
    def add_liquidity(self,amounts,min_mint_amount=self.min_mint_amount):
        pass

    # @TODO: need to implement stableswap pool remove_liquidity_imbalance
    def remove_liquidity_imbalance(self,amounts,max_burn_amount=self.max_burn_amount):
        pass

    # @TODO: update references to actual function names
    # @TODO: track block_timestamp, added as a parameter
    def provide(self,amount,block_timestamp):        
        # I is the peg token index, we assume I=1
        amounts = [None,amount]
        # @TODO: need to implement stableswap pool add_liquidity
        self.add_liquidity(amounts, 0)
        self.last_change = block_timestamp
        # @TODO: need to confirm whether this is user debt or pool debt
        self.debt += amount
        pass

    # @TODO: update refereneces to actual function names
    def withdraw(self,amount,block_timestamp):
        amount = min(amount, self.debt)
        amounts = (None,amount)
        self.remove_liquidity_imbalance(amounts,sys.maxsize)
        self.last_change = block_timestamp
        # @TODO: need to confirm whether this is user debt or pool debt
        self.debt -= amount

    def update(self,beneficiary,block_timestamp):
        if self.last_change + self.ACTION_DELAY > block_timestamp:
            return 0
        # @TODO: finish converting references to POOL
        # balance_pegged = POOL.balances(I)
        # balance_peg = POOL.balances(1 - I) * self.PEG_MUL
        # initial_profit = self._calc_profit()
        # if balance_peg > balance_pegged:
        #      assert self.regulator.provide_allowed(), "Regulator ban"
        #      self._provide(unsafe_sub(balance_peg, balance_pegged) / 5)  # this dumps stablecoin
        # else:
        #     assert self.regulator.withdraw_allowed(), "Regulator ban"
        #     self._withdraw(unsafe_sub(balance_pegged, balance_peg) / 5)  # this pumps stablecoin
        # new_profit: uint256 = self._calc_profit()
        # assert new_profit >= initial_profit, "peg unprofitable"
        # lp_amount: uint256 = new_profit - initial_profit
        # caller_profit: uint256 = lp_amount * self.caller_share / SHARE_PRECISION
        # if caller_profit > 0:
        #     POOL.transfer(_beneficiary, caller_profit)
        # return caller_profit

        pass

    # @TODO: need to add a profit calc
    def calc_profit(self):
        pass

    def withdraw_profit(self,beneficiary):
        lp_amount = self.calc_profit()
        # @TODO: finish converting references to POOL
        # POOL.transfer(self.receiver, lp_amount)
        return lp_amount

    ## PegKeeperRegulator functions

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