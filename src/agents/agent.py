"""Provides the base `Agent` class."""
from abc import ABC


# pylint: disable=too-few-public-methods
class Agent(ABC):
    """Base class for agents."""

    _profit: float = 0.0
    _count: int = 0

    @property
    def profit(self) -> float:
        """Return the profit."""
        return self._profit

    @property
    def count(self) -> int:
        """Return the count."""
        return self._count
