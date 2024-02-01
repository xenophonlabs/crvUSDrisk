"""
Provides a testing suite for the Borrower agent.
"""
from copy import deepcopy
from src.sim import Scenario
from src.agents.borrower import clean, Borrower
from src.sim.utils import clear_controller


def test_clean(scenario: Scenario) -> None:
    """
    Make sure the clean function works as expected.
    """
    controller = deepcopy(scenario.controllers[0])

    kde = scenario.kde[controller.AMM.address]
    debt_log, collateral_log, n = kde.resample(1).T[0]
    debt, collateral, n = clean(debt_log, collateral_log, n)

    assert isinstance(n, int)
    assert isinstance(debt, int)
    assert isinstance(collateral, int)

    assert 4 <= n <= 50
    assert 0 < debt
    assert 0 < collateral


def test_create_loan(scenario: Scenario) -> None:
    """
    The borrower is very simple, just need
    to sanity check that the positions being
    sampled from the KDE are valid.
    """
    borrower = Borrower()
    controller = deepcopy(scenario.controllers[0])
    clear_controller(controller)
    kde = scenario.kde[controller.AMM.address]

    assert borrower.create_loan(controller, kde)
    assert controller.n_loans == 1
    assert borrower.address in controller.loan

    loan = controller.loan[borrower.address]
    assert loan.initial_collateral == borrower.collateral
    assert loan.initial_debt == borrower.debt

    n1, n2 = controller.AMM.read_user_tick_numbers(borrower.address)
    n = n2 - n1 + 1
    assert n == borrower.n
