"""
Provides the baseline scenario strategy.
"""
from typing import List, Type
from ...scenario import Scenario
from ...processing import SingleSimProcessor
from ...results import SingleSimResults
from ....metrics import Metric
from ....logging import get_logger

logger = get_logger(__name__)


# pylint: disable=too-few-public-methods
class BaselineStrategy:
    """
    Strategy for simulating the baseline scenario.
    """

    def __init__(self, metrics: List[Type[Metric]]):
        self.metrics = metrics

    def __call__(
        self,
        scenario: Scenario,
        parameters: dict,
        i: int | None = None,
    ) -> SingleSimResults:
        """
        Takes in a mutable scenario object and executes
        the baseline risk strategy on it.
        """
        logger.info("STARTING new simulation %d", i)

        scenario.generate_pricepaths()  # produce new stochastic prices

        processor = SingleSimProcessor(scenario, self.metrics)

        for sample in scenario.pricepaths:
            scenario.prepare_for_trades(sample)
            scenario.perform_actions(sample)
            # scenario.after_trades()
            processor.update(sample.timestamp, inplace=True)

        logger.info("DONE with simulation %d", i)

        return processor.process()
