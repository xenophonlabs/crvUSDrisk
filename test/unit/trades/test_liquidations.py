"""
Provides a testing suite for liquidations.
"""
from copy import deepcopy
import math
from src.sim import Scenario
from src.trades import Liquidation
from ...utils import approx


# pylint: disable=too-many-locals
def test_liquidation(scenario: Scenario) -> None:
    """
    Test that the liquidation results in the
    expected token transfers.
    """
    _scenario = deepcopy(scenario)
    _scenario.resample_debt()  # this should usually result in some underwater positions

    to_liquidate = {c: c.users_to_liquidate() for c in _scenario.controllers}
    n = sum(len(v) for v in to_liquidate.values())
    assert n > 0

    liquidator = _scenario.liquidator
    liquidator.tolerance = -math.inf

    for controller, positions in to_liquidate.items():
        if len(positions) == 0:
            continue

        # Just look at first position
        position = positions[0]
        total_debt = controller.total_debt()
        bands_x, bands_y = sum(controller.AMM.bands_x.values()), sum(
            controller.AMM.bands_y.values()
        )

        debt = position.debt
        x, y = controller.AMM.get_sum_xy(position.user)
        to_repay = int(controller.tokens_to_liquidate(position.user))
        liquidation = Liquidation(controller, position, to_repay)
        amt_out, _ = liquidation.execute(to_repay)

        assert position.user not in controller.loan
        assert approx(controller.total_debt(), total_debt - debt)
        assert approx(sum(controller.AMM.bands_x.values()), bands_x - x)
        assert approx(sum(controller.AMM.bands_y.values()), bands_y - y)
        assert approx(amt_out, y)

        break
