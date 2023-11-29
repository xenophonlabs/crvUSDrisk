from abc import ABC
from typing import Union
from contextlib import nullcontext
from dataclasses import dataclass
from crvusdsim.pool.crvusd.controller import Position
from curvesim.pool.sim_interface import SimCurvePool
from crvusdsim.pool.sim_interface import SimLLAMMAPool, SimController
from crvusdsim.pool.sim_interface.sim_stableswap import SimCurveStableSwapPool
from crvusdsim.pool.sim_interface.sim_controller import DEFAULT_LIQUIDATOR
from ..modules import ExternalMarket


@dataclass
class Trade(ABC):
    def get_address(self, index: int):
        raise NotImplementedError

    def do(self, use_snapshot_context=False):
        raise NotImplementedError

    def get_decimals(self, index: int):
        raise NotImplementedError


@dataclass
class Swap(Trade):
    pool: Union[
        ExternalMarket,
        SimCurvePool,
        SimCurveStableSwapPool,
        SimLLAMMAPool,
    ]
    i: int
    j: int
    amt: Union[int, float]

    def get_address(self, index: int):
        if isinstance(self.pool, SimCurveStableSwapPool):
            return self.pool.coins[index].address
        return self.pool.coin_addresses[index]

    def get_decimals(self, index: int):
        return self.pool.coin_decimals[index]

    def do(self, use_snapshot_context: bool = False) -> int:
        pool = self.pool
        amt_in = self.amt

        context_manager = (
            pool.use_snapshot_context()
            if use_snapshot_context and not isinstance(pool, ExternalMarket)
            else nullcontext()
        )

        with context_manager:
            # TODO `trade` has different returns for different
            # pool types.
            # TODO for LLAMMA, need to adjust `amt_in` by `in_amount_done`.
            amt_out = pool.trade(self.i, self.j, amt_in)

        return amt_out, self.get_decimals(self.j)


@dataclass
class Liquidation(Trade):
    controller: SimController
    position: Position
    amt: int  # to repay
    frac: float = 10**18  # does nothing
    i: int = 0  # repay stablecoin
    j: int = 1  # receive collateral

    def get_address(self, index: int):
        if index == 0:
            return self.controller.STABLECOIN.address
        else:
            return self.controller.COLLATERAL_TOKEN.address

    def get_decimals(self, index: int):
        if index == 0:
            return self.controller.STABLECOIN.decimals
        else:
            return self.controller.COLLATERAL_TOKEN.decimals

    def do(self, use_snapshot_context=False) -> int:
        """Perform liquidation."""
        context_manager = (
            self.controller.use_snapshot_context()
            if use_snapshot_context
            else nullcontext()
        )

        bal = self.controller.COLLATERAL_TOKEN.balanceOf[DEFAULT_LIQUIDATOR]

        with context_manager:
            self.controller.liquidate_sim(self.position)

        new_bal = self.controller.COLLATERAL_TOKEN.balanceOf[DEFAULT_LIQUIDATOR]

        amt_out = new_bal - bal
        assert amt_out > 0

        return amt_out, self.get_decimals(self.j)
