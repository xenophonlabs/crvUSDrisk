from abc import ABC, abstractmethod
from decimal import Decimal
from .aggregator import AggregateStablePrice
from curvesim.pool.stableswap import CurvePool
import numpy as np

# TODO move to config
PRECISION = 1e18
# PROFIT_THRESHOLD = (
#     1  # FIXME I'm not sure why this is used, but let's err on the side of keeping it
# )
PROFIT_THRESHOLD = 1
CRVUSD_ADDRESS = "0xf939E0A03FB07F59A73314E73794Be0E57ac1b4E"


class PegKeeper(ABC):
    """
    @dev Parent/Abstract class for PegKeeperV1 and PegKeeperV2
    """

    __slots__ = (
        # === Dependencies === #
        "pool",  # pool object
        # === Parameters === #
        "I",  # crvUSD index in pool
        "caller_share",  # share of profits for caller
        "action_delay",  # min delay between actions
        "stabilization_coef",  # smoothing coefficient for stabilizing pool
        "ceiling",  # debt ceiling
        # === State Variables === #
        "debt",  # track minted crvUSD
        "last_change",  # timestamp of last action
        "lp_balance",  # track LP balance
        "precisions",  # precision of each token
    )

    @abstractmethod
    def __init__(self):
        pass

    # === Properties === #

    @property
    def name(self):
        """
        Tokens in underlying pool.
        """
        return self.pool.metadata['name'].replace("Curve.fi Factory Plain Pool: ", "")

    @property
    def profit(self):
        """
        @notice calc profit from LP shares owned by PK.
        @return profit in units of LP tokens
        NOTE assumes both tokens are pegged!
        """
        virtual_price = self.pool.get_virtual_price() / PRECISION
        lp_debt = self.debt / virtual_price + PROFIT_THRESHOLD
        return self.lp_balance - lp_debt  # TODO in contract they floor at 0, why?

    @property
    def left_to_mint(self):
        """
        @notice get the amount of crvUSD this PK is still allowed to mint
        @return amount of crvUSD left to mint
        """
        return self.ceiling - self.debt

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

        assert self.update_allowed(balance_peg, balance_pegged, ts), ValueError(
            "Update not allowed"
        )

        change = self.calc_change(balance_peg, balance_pegged)

        if change == 0:
            return 0

        if balance_peg > balance_pegged:
            self.provide(change)  # this dumps stablecoin
        else:
            self.withdraw(change)  # this pumps stablecoin

        new_profit = self.profit  # new profit since self.debt and self.lp_balance have been updated
        caller_profit = (new_profit - initial_profit) * self.caller_share
        assert (
            new_profit >= initial_profit
        ), "Update unprofitable"  # NOTE if this fails, state (incl pool) is corrupted

        self.last_change = ts

        return caller_profit

    def provide(self, amount: float) -> None:
        """
        @notice Update StableSwap pool object with new balances
        @param amount amount to deposit into stableswap pool
        """
        assert amount > 0, ValueError("Must provide positive amount")

        amount = min(
            amount, self.left_to_mint
        )  # Can deposit at most the amount left to mint
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
        """
        assert amount < 0, ValueError("Must withdraw negative amount")

        amounts = np.zeros(2)
        amounts[self.I] = self.precise(-1*amount, self.I) # make positive

        burned, _ = self.pool.remove_liquidity_imbalance(amounts)

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
        # TODO delete this, calc_change does it now
        # if amount < 0:
        #     amount = -1 * min(
        #         -1 * amount, self.debt
        #     )  # Can withdraw at most the outstanding debt
        # else:
        #     amount = min(
        #         amount, self.left_to_mint
        #     )  # Can deposit at most the amount left to mint

        if amount == 0:
            return 0
            
        amounts = np.zeros(2)
        amounts[self.I] = amount

        lp_balance_diff = self.pool.calc_token_amount(
            amounts
        ) / PRECISION # not accounting for fees

        lp_balance = self.lp_balance + lp_balance_diff
        debt = self.debt + amount

        assert lp_balance >= 0
        assert debt >= 0

        virtual_price = self.pool.get_virtual_price() / PRECISION
        lp_debt = debt / virtual_price + PROFIT_THRESHOLD

        return lp_balance - lp_debt  # TODO in contract they floor at 0, why?

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

        new_profit = self.calc_future_profit(
            self.calc_change(balance_peg, balance_pegged)
        )

        if new_profit < initial_profit:
            # update can only be called if its profitable
            return 0

        return (new_profit - initial_profit) * self.caller_share

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
        amount = (balance_peg - balance_pegged) * self.stabilization_coef
        if amount < 0:
            # withdraw
            amount = -1 * min(
                -1 * amount, self.debt
            )  # Can withdraw at most the outstanding debt
        else:
            # deposit
            amount = min(
                amount, self.left_to_mint
            )  # Can deposit at most the amount left to mint
        return amount

    def precise(self, amount: float, i: int) -> int:
        """
        @notice convert a float to a precise integer for interacting with
        curvesim.
        @param amount amount to convert
        @param i index of token
        @return integer with requisite precision
        FIXME this is useless now since CurveSimPools are normalized
        """
        return int(Decimal(amount) * Decimal(self.precisions[i]))

    @abstractmethod
    def update_allowed(self, balance_peg, balance_pegged, ts):
        pass
