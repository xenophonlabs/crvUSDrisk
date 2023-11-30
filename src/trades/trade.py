import logging
from abc import ABC
from typing import Optional, Tuple
from contextlib import nullcontext
from dataclasses import dataclass
from crvusdsim.pool.crvusd.controller import Position
from curvesim.pool.sim_interface import SimCurvePool
from crvusdsim.pool.sim_interface import SimLLAMMAPool, SimController
from crvusdsim.pool.sim_interface.sim_stableswap import SimCurveStableSwapPool
from crvusdsim.pool.sim_interface.sim_controller import DEFAULT_LIQUIDATOR
from ..modules import ExternalMarket
from ..typing import SimPoolType


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
    pool: SimPoolType
    i: int
    j: int
    amt: Optional[int]

    def get_address(self, index: int):
        if isinstance(self.pool, SimCurveStableSwapPool):
            return self.pool.coins[index].address.lower()
        return self.pool.coin_addresses[index].lower()

    def get_decimals(self, index: int):
        return self.pool.coin_decimals[index]

    def do(self, use_snapshot_context: bool = False) -> Tuple[int, int]:
        pool = self.pool
        amt_in = self.amt

        context_manager = (
            pool.use_snapshot_context()
            if use_snapshot_context and not isinstance(pool, ExternalMarket)
            else nullcontext()
        )

        with context_manager:
            result = pool.trade(self.i, self.j, amt_in)

        if isinstance(pool, ExternalMarket):
            amt_out = result
        elif isinstance(pool, (SimLLAMMAPool, SimCurveStableSwapPool)):
            # TODO for LLAMMA, need to adjust `amt_in` by `in_amount_done`.
            in_amount_done, amt_out, _ = result
            if in_amount_done != amt_in:
                logging.warning(
                    f"LLAMMA amt_in {amt_in} != in_amount_done {in_amount_done}."
                )
        elif isinstance(pool, SimCurvePool):
            amt_out, _ = result
        else:
            raise NotImplementedError

        return amt_out, self.get_decimals(self.j)

    def __repr__(self):
        return f"Swap(pool={self.pool.name}, in={self.pool.coin_names[self.i]}, out={self.pool.coin_names[self.j]}, amt={self.amt})"


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
            return self.controller.STABLECOIN.address.lower()
        else:
            return self.controller.COLLATERAL_TOKEN.address.lower()

    def get_decimals(self, index: int):
        if index == 0:
            return self.controller.STABLECOIN.decimals
        else:
            return self.controller.COLLATERAL_TOKEN.decimals

    def do(self, use_snapshot_context=False) -> Tuple[int, int]:
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
