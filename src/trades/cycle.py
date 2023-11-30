import logging
from typing import List, Union
from scipy.optimize import minimize_scalar
from .trade import Swap, Liquidation

TOLERANCE = 1e-6


class Cycle:
    def __init__(
        self, trades: List[Union[Swap, Liquidation]], expected_profit: float = None
    ):
        self.trades = trades
        self.n = len(trades)
        self.expected_profit = expected_profit
        self.basis_address = trades[0].get_address(trades[0].i)

        # check that this is a cycle
        for i, trade in enumerate(trades):
            if i != self.n - 1:
                next_trade = trades[i + 1]
            else:
                next_trade = trades[0]
            token_out = trade.get_address(trade.j)
            token_in = next_trade.get_address(next_trade.i)
            assert token_in == token_out, (
                token_in,
                token_out,
            )  # "Trades do not form a cycle."

    def execute(self) -> float:
        """Execute trades."""
        logging.info(f"Executing cycle {self}.")
        trade = self.trades[0]
        amt_in = trade.amt

        for i, trade in enumerate(self.trades):
            logging.info(f"Executing trade {trade}.")
            if i != self.n - 1:
                amt_out, decimals = trade.do()
                assert (
                    abs(amt_out - self.trades[i + 1].amt) / 10**decimals < TOLERANCE
                ), f"Trade {i+1} output {amt_out} != Trade {i+2} input {self.trades[i+1].amt}."
            else:
                amt_out, decimals = trade.do()

        profit = (amt_out - amt_in) / 10**decimals
        if abs(profit - self.expected_profit) > TOLERANCE:
            logging.warning(
                f"Expected profit {self.expected_profit} != actual profit {profit}."
            )

        return profit

    def optimize(self):
        """
        Optimize the amt_in for the first trade in the cycle.
        """
        assert all(
            isinstance(trade, Swap) for trade in self.trades
        ), NotImplementedError("Can only optimize swap cycles.")

        trade = self.trades[0]
        high = float(trade.pool.get_max_trade_size(trade.i, trade.j))

        res = minimize_scalar(
            lambda x: -self.populate(x),
            args=(),
            bounds=(0, high),
            method="bounded",
        )

        if res.success:
            self.populate(res.x)
        else:
            raise RuntimeError(res.message)

    def populate(self, amt_in: float) -> float:
        """
        Populate the amt_in to all trades in cycle, and the expected_profit.

        Parameters
        ----------
        amt_in : float
            The amount of the first trade in the cycle.

        Returns
        -------
        expected_profit : float
            The expected profit of the cycle.
        """
        amt_in = int(amt_in)  # cast float to int
        # TODO add a unit test to ensure snapshot context is used correctly

        trade = self.trades[0]
        trade.amt = amt_in

        for i, trade in enumerate(self.trades):
            amt, decimals = trade.do(use_snapshot_context=True)
            if i != self.n - 1:
                self.trades[i + 1].amt = amt

        self.expected_profit = float((amt - amt_in) / 10**decimals)
        return self.expected_profit

    def __repr__(self):
        return f"Cycle(Trades: {self.trades}, Expected Profit: {self.expected_profit})"
