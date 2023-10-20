from decimal import Decimal
from .aggregator import AggregateStablePrice
from curvesim.pool.stableswap import CurvePool
import numpy as np

PRECISION = 1e18
PROFIT_THRESHOLD = 1 # I'm not sure why this is used, but let's err on the side of keeping it
CRVUSD_ADDRESS = '0xf939E0A03FB07F59A73314E73794Be0E57ac1b4E' # TODO move to config

class PegKeeperV1:

    __slots__ = (
        'pool', # pool object
        'I', # crvUSD index in pool
        'caller_share', # share of profits for caller
        'aggregator', # aggregator object
        'action_delay', # min delay between actions
        'stabilization_coef', # smoothing coefficient for stabilizing pool

        # === State Variables === #
        'debt', # track minted crvUSD
        'last_change', # timestamp of last action
        'lp_balance', # track LP balance
    )

    # TODO because we are using curvesim here, we might need to account for smart contract precision
    # TODO we definitely get some floating point rounding errors when multiplying by PRECISION
    def __init__(
            self,
            pool: CurvePool,
            caller_share: float,
            aggregator: AggregateStablePrice,
            action_delay: int = 15 * 60,
            stabilization_coef: float = 0.2,
        ) -> None:


        self.pool = pool
        self.caller_share = caller_share
        self.aggregator = aggregator
        self.action_delay = action_delay
        self.stabilization_coef = stabilization_coef

        self.I = pool.metadata['coins']['addresses'].index(CRVUSD_ADDRESS)
        assert self.I == 1, ValueError('All PK pools should have index==1')

        self.precisions = self.pool.metadata['coins']['decimals']

        # Custom attributes
        # TODO need to incorporate non-zero debt and lp_balance at initialization
        self.debt = 0
        self.last_change = None
        self.lp_balance = 0 

    # === Properties === #
    
    @property
    def profit(self):
        """
        @notice calc profit from LP shares owned by PK.
        @return profit in units of LP tokens
        NOTE assumes both tokens are pegged!
        """
        virtual_price = self.pool.get_virtual_price() / PRECISION
        lp_debt = self.debt / virtual_price + PROFIT_THRESHOLD
        return self.lp_balance - lp_debt # TODO in contract they floor at 0, why?

    # === Main Functions === #

    def update(self, ts: int) -> float:
        """
        @notice Update the pool to maintain peg. Either deposit to lower price
        or withdraw to raise price. Can only be called if profitable, and is the 
        only action performed by the Peg Keeper.
        @param ts timestamp to update at
        @return profit in LP tokens to caller
        NOTE Peg Keeper state is changed inside provide() and withdraw()
        """
        initial_profit = self.profit
        balance_pegged = self.pool.balances[self.I] / self.precisions[self.I]
        balance_peg = self.pool.balances[1 - self.I] / self.precisions[1 - self.I]

        assert self.update_allowed(balance_peg, balance_pegged, ts), ValueError("Update not allowed")

        change = self.calc_change(balance_peg, balance_pegged)

        if balance_peg > balance_pegged:
             self.provide(change)  # this dumps stablecoin
        else:
            self.withdraw(change)  # this pumps stablecoin

        new_profit = self.profit # new profit since self.debt and self.lp_balance have been updated
        caller_profit = (new_profit - initial_profit) * self.caller_share
        assert new_profit >= initial_profit, "Update unprofitable" # NOTE if this fails, state (incl pool) is corrupted

        self.last_change = ts

        return caller_profit

    def provide(self, amount: float) -> None:
        """
        @notice Update StableSwap pool object with new balances
        @param amount amount to deposit into stableswap pool
        NOTE Changes PegKeeper state.
        """
        assert amount > 0, ValueError("Must provide positive amount")

        amounts = np.zeros(2)
        amounts[self.I] = self.precise(amount, self.I)

        minted = self.pool.add_liquidity(amounts) / PRECISION

        # Update state variables
        self.lp_balance += minted
        self.debt += amount

    def withdraw(self, amount: float) -> None:
        """
        @notice Update StableSwap pool object with new balances
        @param amount amount to withdraw from stableswap pool
        NOTE does not change PegKeeper state. This is done in update()
        """
        assert amount < 0, ValueError("Must withdraw negative amount")

        amount = min(-amount, self.debt)
        amounts = np.zeros(2)
        amounts[self.I] = self.precise(amount, self.I)

        burned = self.pool.remove_liquidity_imbalance(amounts, 2**256-1)

        # Update state variables
        self.lp_balance -= burned / PRECISION
        self.debt -= amount

    # === Helpers === #
    
    def calc_future_profit(
            self, 
            amount: float, 
        ) -> float:
        """
        @notice calculate change in LP token profit.
        @param amount amount of token being deposited (positive) or removed (negative)
        @return future profit
        """
        if amount < 0:
            amount = min(amount, self.debt) # Can withdraw at most the outstanding debt

        amounts = np.zeros(2)
        amounts[self.I] = amount

        lp_balance_diff = self.pool.calc_token_amount(amounts) # not accounting for fees

        lp_balance = self.lp_balance + lp_balance_diff
        debt = self.debt + amount

        assert lp_balance >= 0
        assert debt >= 0

        virtual_price = self.pool.get_virtual_price() / PRECISION
        lp_debt = debt / virtual_price + PROFIT_THRESHOLD

        return lp_balance - lp_debt # TODO in contract they floor at 0, why?
    
    def estimate_caller_profit(self, ts: int) -> float:
        """
        @notice Estimate the profit (in LP tokens) of the caller
        if they were to call update() at the given timestamp.
        @param ts timestamp to estimate profit at
        @return profit in LP tokens
        """
        balance_pegged = self.pool.balances[self.I] / self.precisions[self.I]
        balance_peg = self.pool.balances[1 - self.I] / self.precisions[1 - self.I]

        initial_profit = self.profit

        if not self.update_allowed(balance_peg, balance_pegged, ts):
            return 0
        
        new_profit = self.calc_future_profit(self.calc_change(balance_peg, balance_pegged))
        
        if new_profit < initial_profit:
            # update can only be called if its profitable
            return 0

        return (new_profit - initial_profit) * self.caller_share

    def update_allowed(self, balance_peg, balance_pegged, ts):
        """
        @notice check if update is allowed
        @param balance_peg amount of PK token in pool
        @param balance_pegged amount of crvUSD in pool
        @return True if update is allowed, False otherwise
        """
        if self.last_change and self.last_change + self.action_delay > ts:
            return False
        
        p_agg = self.aggregator.price() # crvUSD/USD price from Aggregator

        if balance_peg == balance_pegged:
            return False

        elif balance_peg > balance_pegged:
            # less crvSUD -> crvUSD price above 1 -> deposit more crvUSD
            if p_agg < 1:
                # this pool is off-sync with other pools in aggregator
                return False
        
        else:
            # more crvUSD -> crvUSD price below 1 -> withdraw crvUSD
            if p_agg > 1:
                # this pool is off-sync with other pools in aggregator
                return False

        return True

    def calc_change(
            self, 
            balance_peg: float, 
            balance_pegged: float,
        ) -> float:
        """
        @notice calculate amount of crvUSD to mint or deposit
        @param balance_peg amount of PK token in pool
        @param balance_pegged amount of crvUSD in pool
        @return amount of crvUSD to mint (positive) or deposit (negative)
        """
        return (balance_peg - balance_pegged) * self.stabilization_coef

    def precise(
            self, 
            amount: float, 
            i: int
        ) -> int:
        """
        @notice convert a float to a precise integer for interacting with
        curvesim.
        @param amount amount to convert
        @param i index of token
        @return integer with requisite precision
        """
        return int(Decimal(amount) * Decimal(self.precision[i]))
