"""Provides the `Borrower` class."""
from __future__ import annotations
from typing import Tuple
import secrets
import numpy as np
from scipy.stats import gaussian_kde
from crvusdsim.pool.sim_interface import SimController
from .agent import Agent
from ..logging import get_logger

logger = get_logger(__name__)

TOLERANCE = 1e-4


def clean(
    debt_log: float, collateral_log: float, n: int, decimals: int = 18
) -> Tuple[int, int, int]:
    """
    Process log values into integers with the right decimals.
    """
    debt = int(np.exp(debt_log) * 1e18)
    collateral = int(np.exp(collateral_log) * 10**decimals)  # TODO decimals
    n = min(max(int(n), 4), 50)
    return debt, collateral, n


# pylint: disable=too-few-public-methods
class Borrower(Agent):
    """
    The Borrower either deposits or repays crvusd
    positions in the Controller.
    """

    def __init__(self) -> None:
        super().__init__()
        self.address = "0x" + secrets.token_hex(20)

    def create_loan(self, controller: SimController, kde: gaussian_kde) -> bool:
        """
        Create a loan in the given controller, sampled from
        the input KDE.

        The KDE is trained on (log(debt), log(collateral), n)
        """
        debt_log, collateral_log, n = kde.resample(1).T[0]
        debt, collateral, n = clean(
            debt_log, collateral_log, n, decimals=controller.COLLATERAL_TOKEN.decimals
        )
        controller.COLLATERAL_TOKEN._mint(  # pylint: disable=protected-access
            self.address, collateral
        )

        try:
            controller.health_calculator(self.address, collateral, debt, False, n)
            controller.create_loan(self.address, collateral, debt, n)
        except AssertionError as e:
            if str(e) == "Debt too high":
                debt = controller.max_borrowable(collateral, n)
                if debt == 0:
                    logger.warning(
                        "Controller %s won't accept any more debt.", controller.AMM.name
                    )
                    del controller.loan[self.address]  # remove the key
                    return False
                controller.create_loan(self.address, collateral, debt, n)
            else:
                raise e

        return True
