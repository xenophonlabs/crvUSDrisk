from .pegkeeper import PegKeeper
from .pegkeeperregulator import PegKeeperRegulator
from curvesim.pool.stableswap import CurvePool

import numpy as np

# TODO move to config
PRECISION = 1e18
PROFIT_THRESHOLD = (
    1  # I'm not sure why this is used, but let's err on the side of keeping it
)
CRVUSD_ADDRESS = "0xf939E0A03FB07F59A73314E73794Be0E57ac1b4E"


class PegKeeperV2(PegKeeper):
    __slots__ = PegKeeper.__slots__ + ("regulator",)  # regulator object

    def __init__(
        self,
        pool: CurvePool,
        regulator: PegKeeperRegulator,
        caller_share: float,
        ceiling: float,
        action_delay: float = 15 * 60,
        stabilization_coef: float = 0.2,
    ) -> None:
        self.pool = pool
        self.regulator = regulator
        self.caller_share = caller_share
        self.action_delay = action_delay
        self.stabilization_coef = stabilization_coef
        self.ceiling = ceiling

        # Precision tracking
        self.precisions = self.pool.metadata["coins"]["decimals"]
        self.I = pool.metadata["coins"]["addresses"].index(CRVUSD_ADDRESS)
        assert self.I == 1, ValueError("All PK pools should have index==1")

        # TODO need to incorporate non-zero debt and lp_balance at initialization
        self.debt = 0
        self.last_change = None
        self.lp_balance = 0

    def update_allowed(self, balance_peg, balance_pegged, ts):
        """
        @notice check if update is allowed
        @param balance_peg amount of PK token in pool
        @param balance_pegged amount of crvUSD in pool
        @return True if update is allowed, False otherwise
        """
        if self.last_change and self.last_change + self.action_delay > ts:
            return False

        if balance_peg == balance_pegged:
            return False

        elif balance_peg > balance_pegged:
            # less crvSUD -> crvUSD price above 1 -> deposit more crvUSD
            return self.regulator.provide_allowed()

        else:
            # more crvUSD -> crvUSD price below 1 -> withdraw crvUSD
            return self.regulator.withdraw_allowed()
