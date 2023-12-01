"""Provides the `Borrower` class."""
from .agent import Agent


# pylint: disable=too-few-public-methods
class Borrower(Agent):
    """
    The Borrower either deposits or repays crvusd
    positions in the Controller.
    """
