"""
Provides the baseline scenario strategy.
"""
from typing import List, Type
from ...scenario import Scenario
from ...processing import SingleSimProcessor
from ...results import SingleSimResults
from ....metrics import Metric


# pylint: disable=too-few-public-methods
class BaselineStrategy:
    # TODO base strategy?
    """
    Strategy for simulating the baseline scenario.
    """

    def __init__(self, metrics: List[Type[Metric]]):
        self.metrics = metrics

    def __call__(self, scenario: Scenario, parameters: dict) -> SingleSimResults:
        """
        Takes in a mutable scenario object and executes
        the baseline risk strategy on it.
        """
        scenario.generate_pricepaths()  # produce new stochastic prices
        scenario.prepare_for_run()

        processor = SingleSimProcessor(scenario, self.metrics)

        for sample in scenario.pricepaths:
            scenario.prepare_for_trades(sample)
            scenario.perform_actions(sample)
            # scenario.after_trades()
            processor.update(sample.timestamp, inplace=True)

        return processor.process()
