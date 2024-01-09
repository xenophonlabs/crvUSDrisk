"""Provides the base `Agent` class."""
from __future__ import annotations
from abc import ABC
from functools import cached_property
from typing import TYPE_CHECKING
from ..logging import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..prices import PriceSample
    from ..trades import Cycle


TOLERANCE = -10  # USD


# pylint: disable=too-few-public-methods
class Agent(ABC):
    """Base class for agents."""

    _profit: float = 0.0
    _count: int = 0
    _volume: int = 0
    _borrower_loss: float = 0.0

    @property
    def profit(self) -> float:
        """Return the profit."""
        return self._profit

    @property
    def count(self) -> int:
        """Return the count."""
        return self._count

    @property
    def volume(self) -> int:
        """Return the volume."""
        return self._volume

    @property
    def borrower_loss(self) -> float:
        """Return the borrower loss."""
        return self._borrower_loss

    @cached_property
    def name(self) -> str:
        """Agent name."""
        return type(self).__name__

    def update_borrower_losses(self, cycle: Cycle, prices: PriceSample) -> None:
        """
        Update the borrower loss. Applicable to
        the Liquidator and Arbitrageur child classes.
        """
        for i in cycle.llamma_trades:
            trade = cycle.trades[i]
            token_in = (
                trade.amt
                / 10 ** trade.get_decimals(trade.i)
                * prices.prices_usd[trade.get_address(trade.i)]
            )
            token_out = (
                trade.out
                / 10 ** trade.get_decimals(trade.j)
                * prices.prices_usd[trade.get_address(trade.j)]
            )
            borrower_loss = token_out - token_in
            if borrower_loss < TOLERANCE:
                logger.debug(
                    "Borrower loss was positive for cycle %s: %f", cycle, borrower_loss
                )
            self._borrower_loss += borrower_loss
