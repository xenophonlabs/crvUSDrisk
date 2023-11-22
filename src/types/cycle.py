from typing import List, Union
from dataclasses import dataclass
from ..modules.market import ExternalMarket
from curvesim.pool.sim_interface import SimCurvePool
from crvusdsim.pool.sim_interface import SimLLAMMAPool, SimController
from crvusdsim.pool.sim_interface.sim_stableswap import SimCurveStableSwapPool
from crvusdsim.pool.sim_interface.sim_controller import DEFAULT_LIQUIDATOR
import logging
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
    price: float = (None,)  # price for External Market

    def get_address(self, index: int):
        if isinstance(self.pool, SimCurveStableSwapPool):
            return self.pool.coins[index].address
        return self.pool.coin_addresses[index]

    def get_decimals(self, index: int):
        if isinstance(self.pool, SimCurveStableSwapPool):
            return self.pool.coins[index].decimals
        elif isinstance(self.pool, ExternalMarket):
            return 0  # TODO is there a better way to handle this?
        return self.pool.coin_decimals[index]

    def do(self, precision=True):
        pool = self.pool
        amt_out = pool.trade(self.i, self.j, self.amt)
        decimals = pool.coin_decimals[self.j]  # 0 for External Mkt

        if not precision:
            amt_out /= 10**decimals

        return amt_out


@dataclass
class Liquidation(Trade):
    controller: SimController
    user: str
    amt: int  # to repay
    frac: float = 10**18
    i: int = 0  # repay stablecoin
    j: int = 1  # receive collateral

    def get_address(self, index: int):
        if index == 0:
            return self.pool.STABLECOIN.address
        else:
            return self.pool.COLLATERAL_TOKEN.address

    def get_decimals(self, index: int):
        if index == 0:
            return self.pool.STABLECOIN.decimals
        else:
            return self.pool.COLLATERAL_TOKEN.decimals

    def do(self, precision=True) -> int:
        """Perform liquidation."""
        # Check change in balance
        bal = self.controller.COLLATERAL_TOKEN.balanceOf[DEFAULT_LIQUIDATOR]
        self.controller.liquidate(DEFAULT_LIQUIDATOR, self.user, 0)
        new_bal = self.controller.COLLATERAL_TOKEN.balanceOf[DEFAULT_LIQUIDATOR]

        amt_out = new_bal - bal
        assert amt_out > 0

        if not precision:
            decimals = self.controller.COLLATERAL_PRECISION
            amt_out /= 10**decimals

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
                amt_out = trade.do(precision=False)
                # check that the amt_out from trade[i] == amt_in for trade[i+1]
                assert (
                    abs(amt_out - self.trades[i + 1].amt) < 1e-6
                ), f"Trade {i} output {amt_out} != Trade {i+1} input {self.trades[i+1].amt}."
            else:
                # Fix decimals to compute profit
                amt_out = trade.do(precision=True)

        profit = amt_out - amt_in
        if abs(profit - self.expected_profit) > 1e-6:
            logging.warning(
                f"Expected profit {self.expected_profit} != actual profit {profit}."
            )

        return profit
