from collections import defaultdict
from .llamma import LLAMMA
from .mpolicy import MonetaryPolicy

DEAD_SHARES = 1e-15 # to init shares in a band

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