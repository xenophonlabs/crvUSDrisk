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

        # check that this is a cycle
        for i, trade in enumerate(trades):
            if i != self.n - 1:
                next_trade = trades[i + 1]
            else:
                next_trade = trades[0]
            token_out = trade.get_address(trade.j).lower()
            token_in = next_trade.get_address(next_trade.i).lower()
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

    def optimize(self):
        """
        Optimize the amt_in for the first trade in the cycle.
        """
        # TODO maybe the move is to have a use_snapshot_context type thing for
        # the cycle. Then we can string the Swaps together using the snapshot
        # context and verify the profit. We can then just use scipy minimize_scalar.
        trade = self.trades[0]
        high = trade.pool.get_max_trade_size(self.trade.i, self.trade.j)

        res = minimize_scalar(
            self.populate,
            args=(),
            bracket=(0, high),
            xtol=1e-6,
            method="brentq",
        )

        if res.converged:
            self.populate(res.x)
        else:
            raise RuntimeError(res.message)

    def populate(self, amt_in):
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
        # NOTE ensure we are using the snapshot context
        # NOTE ensure we are using the correct decimals
        pass

    def __repr__(self):
        return f"Cycle(Trades: {self.trades}, Expected Profit: {self.expected_profit})"
