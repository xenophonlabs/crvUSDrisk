from collections import defaultdict
import math
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
        return f'Position\nuser={self.user}\nx={self.x}\ny={self.y}\ndebt={self.debt}\nhealth={self.health}'
    
    def __str__(self) -> str:
        return f'Position\nuser={self.user}\nx={self.x}\ny={self.y}\ndebt={self.debt}\nhealth={self.health}'

class Controller:

    # TODO: implement interest rates
    # TODO: each loan should have its own liq disc if it changes in the future

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

        self.loans = defaultdict(int)

    def create(self):
        """
        @notice create positions to approximate a target distribution 
        of collateral/health.
        TODO: a function to open a loan taking as input the target debt (or collateral)
        and a target health. Then we can create loans as a func of a health/collateral
        distribution.
        """
        pass

    def health(self, user):
        """
        @notice simple health computation for user
        @param user user address
        @return health of user
        TODO: missing the get_sum_xy component if full=True on the contract
        """
        health = (self.amm.get_x_down(user) * (1 - self.liquidation_discount)/self.loans[user]) - 1
        return health

    def liquidate(self, user):
        pass

    def deposit(self, user, amount, N):
        pass

    def withdraw(self, user, frac):
        pass

    def repay(self, user, amount):
        pass

    # === Helper Functions === #

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
        for i in range(1, N):
            d_y_effective = d_y_effective * (self.A - 1) / self.A
            y_effective += d_y_effective
        return y_effective
    
    def _calculate_debt_n1(self, collateral, debt, N) -> int:
        """
        @notice Calculate the upper band number for the deposit to sit in to support
                the given debt. Reverts if requested debt is too high.
        @param collateral Amount of collateral (at its native precision)
        @param debt Amount of requested debt
        @param N Number of bands to deposit into
        @return Upper band n1 (n1 <= n2) to deposit into. Signed integer
        """
        assert debt > 0, "No loan"
        n0 = self.amm.active_band()
        p_base = self.amm.p_oracle_up(n0)

        y_effective = self.get_y_effective(collateral, N, self.loan_discount)

        ratio = y_effective * p_base / (debt + EPSILON)

        assert y_effective > 0, "Amount too low"
        n_delta = math.log(y_effective, base=(self.A/(self.A - 1))) 

        n1 = n0 + n_delta
        # if n1 <= n0:
        #     assert self.amm.can_skip_bands(n1 - 1), "Debt too high"

        assert self.amm.p_o_up(n1) < self.amm.p_o(), "Debt too high"

        return n1
