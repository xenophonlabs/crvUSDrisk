from typing import List, Union
from dataclasses import dataclass
from ..modules.market import ExternalMarket
from crvusdsim.pool.crvusd.controller import Position
from curvesim.pool.sim_interface import SimCurvePool
from crvusdsim.pool.sim_interface import SimLLAMMAPool, SimController
from crvusdsim.pool.sim_interface.sim_stableswap import SimCurveStableSwapPool
from crvusdsim.pool.sim_interface.sim_controller import DEFAULT_LIQUIDATOR
import logging
from abc import ABC

TOLERANCE = 1e-6


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


class Cycle:
    def __init__(
        self, trades: List[Union[Swap, Liquidation]], expected_profit: float = None
    ):
        self.trades = trades
        self.n = len(trades)
        self.expected_profit = expected_profit

        # check that this is a cycle
        for i, trade in enumerate(trades):
            if i != self.n - 1:
                next_trade = trades[i + 1]
            else:
                next_trade = trades[0]
            token_out = trade.get_address(trade.j)
            token_in = next_trade.get_address(next_trade.i)
            assert token_in == token_out, "Trades do not form a cycle."

    def execute(self) -> float:
        """Execute trades."""
        trade = self.trades[0]
        amt_in = trade.amt / 10 ** trade.get_decimals(trade.i)

        for i, trade in enumerate(self.trades):
            logging.info(f"Executing trade {trade}.")
            if i != self.n - 1:
                amt_out = trade.do()
                # check that the amt_out from trade[i] == amt_in for trade[i+1]
                assert (
                    abs(amt_out - self.trades[i + 1].amt) / 1e18
                    < TOLERANCE  # TODO precision for ext mkt
                ), f"Trade {i+1} output {amt_out} != Trade {i+2} input {self.trades[i+1].amt}."
            else:
                # Fix decimals to compute profit
                amt_out = trade.do()

        profit = amt_out - amt_in
        if abs(profit - self.expected_profit) / 1e18 > TOLERANCE:
            logging.warning(
                f"Expected profit {self.expected_profit} != actual profit {profit}."
            )

        return profit

    def __repr__(self):
        return f"Cycle(Trades: {self.trades}, Expected Profit: {self.expected_profit})"
