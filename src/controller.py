from collections import defaultdict
import math
from typing import List
import numpy as np
from .llamma import LLAMMA
from .mpolicy import MonetaryPolicy

DEAD_SHARES = 1e-15 # to init shares in a band
EPSILON = 1e-18 # to avoid division by 0

class Position:

    def __init__(self, user, x, y, debt, health):
        self.user = user
        self.x = x
        self.y = y
        self.debt = debt
        self.health = health

    def __repr__(self) -> str:
        return f'Position:(user={self.user},x={self.x},y={self.y},debt={self.debt},health={self.health})'
    
    def __str__(self) -> str:
        return f'Position:(user={self.user},x={self.x},y={self.y},debt={self.debt},health={self.health})'

class Controller:
    """
    @notice A simplified python implementation of the crvUSD Controller with
    enough functionality to model risk in the system.
    TODO implement interest rates
    TODO each loan should have its own liq disc if it changes in the future
    TODO withdraw
    TODO repay
    TODO limit amt of controller debt
    TODO currently to generate the distribution we just create loans but this
    means that we aren't accounting for loans that are in soft liquidation. We
    need a way to generate a distribution which already has loans in soft liquidation?
    """

    __slots__ = (
        # === Parameters === #
        'loan_discount', # defines initial LTV
        'liquidation_discount', # defines liquidation threshold
        'MIN_TICKS', # minimum number of bands for position
        'MAX_TICKS', # maximum number of bands for position
        'A', # A parameter

        # === State variables === #
        'loans', # loans[user] = debt
        'total_debt', # total debt

        # === Dependencies/Inputs === #
        'monetary_policy', # MonetaryPolicy object
        'amm', # LLAMMA object
    )

    def __init__(
            self, 
            amm: LLAMMA, 
            monetary_policy: MonetaryPolicy,
            loan_discount: float,
            liquidation_discount: float,
            MIN_TICKS: int=4,
            MAX_TICKS: int=50,
        ):

        self.A = amm.A
        self.monetary_policy = monetary_policy
        self.amm = amm
        self.loan_discount = loan_discount
        self.liquidation_discount = liquidation_discount
        self.MIN_TICKS = MIN_TICKS
        self.MAX_TICKS = MAX_TICKS
        
        self.total_debt = 0
        self.loans = defaultdict(int)

    def gen_borrowers(
        self,
        n: int,
        coins: float, # target collateral
        mean_N: int = 10,
        std_N: int = 3,
        v: bool = False,
        ):
        """
        @notice generate n borrowers. borrowers are
        a tuple (collateral, debt, N). 
        @param n number of borrowers to generate
        @param mean_N mean of normal distribution for N
        @param std_N std of normal distribution for N
        @return borrowers list of tuples (collateral, debt, N)

        NOTE For now, we create them as:
        1. Generate uniform (dirichlet) collateral distribution from target COINS
        2. Generate normal distribution for N
        For each borrower:
        3. a. calculate max_borrowable given collateral, N
        3. b. generate random riskiness from [0.5, 1] <- TODO this was arbitrary to get us to ~40M debt
        3. c. set debt = max_borrowable * riskiness
        TODO include whales (not just uniform dist of collateral)
        TODO improve riskiness generation
        TODO compare to empirical data
        TODO could replace riskiness with a target health
        """
        borrowers = []
        collateral = np.random.dirichlet(np.ones(n), 1)[0] * coins
        Ns = np.clip(np.random.normal(mean_N, std_N, n), 4, 50).astype(int)
        for i in range(n):
            max_borrowable = self.max_borrowable(collateral[i], Ns[i])
            riskiness = np.random.uniform(low=0.5)
            debt = riskiness * max_borrowable
            borrowers.append((collateral[i], debt, Ns[i]))
        borrowers = np.array(borrowers)
        assert abs(borrowers[:,0].sum() - coins) <= 1e-3
        if v:
            print(f"Total collateral: {round(borrowers[:,0].sum() * self.amm.base_price / 1e6)} Mns USD")
            print(f"Total debt: {round(borrowers[:,1].sum() / 1e6)} Mns USD")
        return borrowers

    def health(self, user):
        """
        @notice simple health computation for user
        @param user user address
        @return health of user
        TODO: missing the get_sum_xy component if full=True on the contract
        """
        health = (self.amm.get_x_down(user) * (1 - self.liquidation_discount)/self.loans[user]) - 1
        return health

    def liquidate(
            self, 
            user: str,
            frac: float,
        ) -> None:
        """
        @notice liquidate a fraction of a user's debt. This is a hard liquidation.
        @param user user address
        @param frac fraction of debt to liquidate
        @return [x_pnl, y_pnl] pnl in crvUSD and collateral of liquidator
        """
        assert self.health(user) < 0, "Not enough rekt"

        debt_initial = self.loans[user]
        debt_liquidated = debt_initial * frac
        debt_final = debt_initial - debt_liquidated
        
        x_liquidated, y_liquidated = self.amm.withdraw(user, frac)

        # delta is the amount of crvUSD leftover from position
        # or is the remaining crvUSD needed to close position
        delta = x_liquidated - debt_liquidated
        x_pnl = delta # liquidator either pockets a positive delta or pays a negative delta
        y_pnl = y_liquidated # liquidator pockets collateral

        if debt_final == 0:
            del self.loans[user]
        else:
            self.loans[user] = debt_final

        self.total_debt -= debt_liquidated

        return x_pnl, y_pnl
    
    def check_liquidate(self, user, frac):
        assert self.health(user) < 0, "Not enough rekt"

        debt_initial = self.loans[user]
        debt_liquidated = debt_initial * frac
        
        x_liquidated, y_liquidated = self.amm.get_sum_xy(user) * frac

        # delta is the amount of crvUSD leftover from position
        # or is the remaining crvUSD needed to close position
        delta = x_liquidated - debt_liquidated
        x_pnl = delta # liquidator either pockets a positive delta or pays a negative delta
        y_pnl = y_liquidated # liquidator pockets collateral

        return x_pnl, y_pnl

    def create_loan(
            self, 
            user: str, 
            collateral: float, 
            debt: float, 
            N: int
        ) -> None:
        assert self.MIN_TICKS <= N <= self.MAX_TICKS, "Invalid number of bands"
        assert self.loans[user] == 0, "User already has a loan"

        n1 = self.calculate_debt_n1(collateral, debt, N)
        n2 = int(n1 + (N - 1))

        self.loans[user] = debt
        self.total_debt += debt
        self.amm.deposit(user, collateral, n1, n2)

    def withdraw(self, user, frac):
        pass

    def repay(self, user, amount):
        pass

    # === Helper Functions === #

    def users_to_liquidate(self) -> List[Position]:
        to_liquidate = []
        for user, debt in self.loans.items():
            if self.health(user) < 0:
                x, y = self.amm.get_sum_xy(user)
                to_liquidate.append(Position(user, x, y, debt, self.health(user)))
        return to_liquidate
    
    def max_borrowable(
            self,
            collateral: float,
            N: int,
        ) -> float:
        """
        @notice compute max debt for a given collateral amount
        TODO using amm.p_o_down(amm.active_band) is not the same 
        as max_p_base() from controller contract.
        """
        return self.get_y_effective(collateral, N) * self.amm.p_o_down(self.amm.active_band)

    def get_y_effective(self, collateral, N) -> float:
        """
        @notice Compute the value of the collateral 
        @param collateral Amount of collateral to get the value for
        @param N Number of bands the deposit is made into
        @param discount Loan discount at 1e18 base (e.g. 1e18 == 100%)
        @return y_effective
        """
        discount = min(self.loan_discount + DEAD_SHARES / max(collateral / N, DEAD_SHARES), 1)
        d_y_effective = collateral / N * (1 - discount) * ((self.A-1)/self.A) ** 0.5
        y_effective = d_y_effective
        for _ in range(1, N):
            d_y_effective = d_y_effective * (self.A - 1) / self.A
            y_effective += d_y_effective
        return y_effective
    
    def calculate_debt_n1(self, collateral, debt, N) -> int:
        """
        @notice Calculate the upper band number for the deposit to sit in to support
                the given debt. Reverts if requested debt is too high.
        @param collateral Amount of collateral (at its native precision)
        @param debt Amount of requested debt
        @param N Number of bands to deposit into
        @return Upper band n1 (n1 <= n2) to deposit into. Signed integer
        """
        if isinstance(N, float):
            N = int(N)
        assert debt > 0, "No loan"
        n0 = self.amm.active_band
        p_base = self.amm.p_o_up(n0)
        y_effective = self.get_y_effective(collateral, N)
        ratio = y_effective * p_base / (debt + EPSILON)

        n_delta = math.ceil(math.log(ratio, self.A/(self.A - 1))) 
        # n_delta = math.ceil(math.log(ratio) / math.log(self.A / (self.A - 1)))

        n1 = n0 + n_delta
        # if n1 <= n0:
        #     assert self.amm.can_skip_bands(n1 - 1), "Debt too high"

        assert self.amm.p_o_up(n1) < self.amm.p_o, "Debt too high"

        return int(n1)
