"""Provides the base `Agent` class."""
from abc import ABC
from functools import cached_property


# pylint: disable=too-few-public-methods
class Agent(ABC):
    """Base class for agents."""

    _profit: float = 0.0
    _count: int = 0
    _volume: int = 0

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
        # TODO implement in each agent
        return self._volume

    @cached_property
    def name(self) -> str:
        """Agent name."""
        return type(self).__name__
