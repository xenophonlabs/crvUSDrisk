"""
Provides metrics on the Agents.
"""

from typing import List, TYPE_CHECKING, Union, Dict
from functools import cached_property
from .base import Metric
from .utils import entity_str

if TYPE_CHECKING:
    from ..agents import Agent


class AgentMetrics(Metric):
    """
    Metrics computed on Agents.
    """

    def __init__(self, **kwargs) -> None:
        self.agents: List[Agent] = kwargs["agents"]

    @cached_property
    def config(self) -> dict:
        summary: Dict[str, List[str]] = {}
        plot: Dict[str, dict] = {}
        for agent in self.agents:
            agent_str = entity_str(agent, "agent")
            summary[agent_str + "_profit"] = ["max"]  # proxy for `last`
            plot[agent_str + "_profit"] = {
                "title": f"{agent.name} Profit",
                "kind": "line",
            }
            summary[agent_str + "_volume"] = ["max"]  # proxy for `last`
            plot[agent_str + "_volume"] = {
                "title": f"{agent.name} Volume",
                "kind": "line",
            }
            summary[agent_str + "_count"] = []

        return {"functions": {"summary": summary}, "plot": plot}

    def compute(self) -> Dict[str, Union[int, float]]:
        res = []
        for agent in self.agents:
            res.append(agent.profit)
            res.append(agent.volume)
            res.append(agent.count)
        return dict(zip(self.cols, res))
