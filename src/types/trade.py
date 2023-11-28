from typing import Union
from dataclasses import dataclass
from ..modules import ExternalMarket
from crvusdsim.pool.crvusd.controller import Position
from curvesim.pool.sim_interface import SimCurvePool
from crvusdsim.pool.sim_interface import SimLLAMMAPool, SimController
from crvusdsim.pool.sim_interface.sim_stableswap import SimCurveStableSwapPool
from crvusdsim.pool.sim_interface.sim_controller import DEFAULT_LIQUIDATOR
from abc import ABC


@dataclass
class Trade(ABC):
    def get_address(self, index: int):
        raise NotImplementedError

    def do(self, precision=True):
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
        # if isinstance(self.pool, SimCurveStableSwapPool):
        #     return self.pool.coins[index].decimals
        return self.pool.coin_decimals[index]

    def do(self):
        pool = self.pool

        amt_in = self.amt
        if isinstance(pool, ExternalMarket):
            # TODO find a better way to handle decimals
            amt_in /= 10 ** self.get_decimals(self.i)

        amt_out = pool.trade(self.i, self.j, amt_in)
        return amt_out


@dataclass
class Liquidation(Trade):
    controller: SimController
    position: Position
    amt: int  # to repay
    frac: float = 10**18
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

    def do(self) -> int:
        """Perform liquidation."""
        # Check change in balance
        bal = self.controller.COLLATERAL_TOKEN.balanceOf[DEFAULT_LIQUIDATOR]
        self.controller.liquidate_sim(self.position)
        new_bal = self.controller.COLLATERAL_TOKEN.balanceOf[DEFAULT_LIQUIDATOR]

        amt_out = new_bal - bal
        assert amt_out > 0
        return amt_out
