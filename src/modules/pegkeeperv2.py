import sys 

class PegKeeper:

    def __init__(
            self,
            pool,
            I,
            caller_share,
            aggregator,
            action_delay = 15 * 60,
        ) -> None:
        
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

        pool_balances = [0,0]
    
    # @TODO: need to implement stableswap pool add_liquidity
    def add_liquidity(self,amounts,min_mint_amount=self.min_mint_amount):
        pass

    # @TODO: need to implement stableswap pool remove_liquidity_imbalance
    def remove_liquidity_imbalance(self,amounts,max_burn_amount=self.max_burn_amount):
        pass

    # @TODO: need to implement stableswap pool transfer
    def pool_transfer(beneficiary, caller_profit):
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
        # NOTE: we added this (assume I=1)
        balance_pegged = self.pool_balances[1]
        # NOTE: we added this, assume I=1, 1-I=0
        balance_peg = self.pool_balances[0] * self.PEG_MUL
        initial_profit = self._calc_profit()
        if balance_peg > balance_pegged:
             assert self.regulator.provide_allowed(), "Regulator ban"
             self._provide((balance_peg - balance_pegged) / 5)  # this dumps stablecoin
        else:
            assert self.regulator.withdraw_allowed(), "Regulator ban"
            self._withdraw((balance_pegged - balance_peg) / 5)  # this pumps stablecoin
        new_profit = self.calc_profit()
        assert new_profit >= initial_profit, "peg unprofitable"
        lp_amount = new_profit - initial_profit
        caller_profit = lp_amount * self.caller_share / self.SHARE_PRECISION
        if caller_profit > 0:
            # NOTE: modified to a function
            self.pool_transfer(beneficiary, caller_profit)
        return caller_profit

        pass

    # @TODO: need to add a profit calc
    def calc_profit(self):
        pass

    def withdraw_profit(self,beneficiary):
        lp_amount = self.calc_profit()
        # @TODO: finish converting references to POOL
        # POOL.transfer(self.receiver, lp_amount)
        return lp_amount

