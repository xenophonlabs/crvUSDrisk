from .pegkeeper import PegKeeper
from .aggregator import AggregateStablePrice
from curvesim.pool.stableswap import CurvePool
import numpy as np

# TODO move to config
PRECISION = 1e18
PROFIT_THRESHOLD = (
    1  # I'm not sure why this is used, but let's err on the side of keeping it
)
CRVUSD_ADDRESS = "0xf939E0A03FB07F59A73314E73794Be0E57ac1b4E"


class PegKeeperV1(PegKeeper):
    __slots__ = PegKeeper.__slots__ + ("aggregator",)  # aggregator object

    def __init__(
        self,
        pool: CurvePool,
        aggregator: AggregateStablePrice,
        caller_share: float,
        ceiling: float,
        action_delay: int = 15 * 60,
        stabilization_coef: float = 0.2,
    ) -> None:
        self.pool = pool
        self.caller_share = caller_share
        self.aggregator = aggregator
        self.action_delay = action_delay
        self.stabilization_coef = stabilization_coef
        self.ceiling = ceiling

        # Precision tracking
        # self.precisions = [10**d for d in self.pool.metadata["coins"]["decimals"]]
        # FIXME now that I am using CurveSimPool instead of CurvePool we already get
        # normalized balances. Check where I am adjusting for precision
        self.precisions = np.ones(2)
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

        p_agg = self.aggregator.price()  # crvUSD/USD price from Aggregator

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
