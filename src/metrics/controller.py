"""
Provides metrics on the Controllers.
"""

from __future__ import annotations
from typing import List, TYPE_CHECKING, Union, Dict
from functools import cached_property
import numpy as np
from .base import Metric
from .utils import entity_str

if TYPE_CHECKING:
    from crvusdsim.pool.sim_interface import SimController


class ControllerMetrics(Metric):
    """
    Metrics computed on the Controller.
    """

    def __init__(self, **kwargs) -> None:
        self.controllers: List[SimController] = kwargs["controllers"]

    @cached_property
    def config(self) -> dict:
        summary: Dict[str, List[str]] = {}
        plot: Dict[str, dict] = {}
        for controller in self.controllers:
            controller_str = entity_str(controller, "controller")
            summary[controller_str + "_system_health"] = ["mean", "min"]
            plot[controller_str + "_system_health"] = {
                "title": f"{controller_str} System Health",
                "kind": "line",
            }
            summary[controller_str + "_bad_debt"] = ["mean", "max"]
            plot[controller_str + "_bad_debt"] = {
                "title": f"{controller_str} Bad Debt",
                "kind": "line",
            }
            summary[controller_str + "_num_loans"] = []
            summary[controller_str + "_total_debt"] = []
            summary[controller_str + "_users_to_liquidate"] = []
        return {"functions": {"summary": summary}, "plot": plot}

    def compute(self) -> Dict[str, Union[int, float]]:
        res: List[Union[int, float]] = []
        for controller in self.controllers:
            healths = ControllerMetrics.controller_healths(controller)
            debts = ControllerMetrics.controller_debts(controller)
            res.append(ControllerMetrics.controller_system_health(healths, debts))
            res.append(ControllerMetrics.controller_bad_debt(healths, debts))
            res.append(controller.n_loans)
            res.append(controller.total_debt() / 1e18)
            res.append(ControllerMetrics.users_to_liquidate(healths))
        return dict(zip(self.cols, res))

    @staticmethod
    def controller_healths(controller: SimController) -> np.ndarray:
        """Return array of healths of controller users."""
        return np.array(
            [controller.health(user, full=True) for user in controller.loan.keys()]
        )

    @staticmethod
    def controller_debts(controller: SimController) -> np.ndarray:
        """Return array of debts of controller users."""
        return np.array([l.initial_debt for l in controller.loan.values()])

    @staticmethod
    def controller_system_health(healths: np.ndarray, debts: np.ndarray) -> float:
        """
        Calculate the system health of a controller.
        We calculate this as a weighted average of user
        health, where weights are each user's initial debt.
        TODO use current debt instead of initial debt <- rate
        """
        return (healths * debts).sum() / debts.sum() / 1e18

    @staticmethod
    def controller_bad_debt(healths: np.ndarray, debts: np.ndarray) -> float:
        """
        Calculate net bad debt in controller.
        We define bad debt as the debt of users with
        health < 0.
        TODO use current debt instead of initial debt <- rate
        """
        users = np.where(healths < 0)
        return debts[users].sum() / 1e18

    @staticmethod
    def users_to_liquidate(healths: np.ndarray) -> int:
        """
        Calculate the number of users to liquidate.
        """
        return len(np.where(healths < 0))
