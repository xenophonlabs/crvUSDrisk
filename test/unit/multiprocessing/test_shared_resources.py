"""
Test that there are no shared resources when we deepcopy
a scenario.
"""
from typing import Any
from copy import deepcopy
from src.sim import Scenario

immutable_types = (int, float, str, tuple, frozenset)
container_types = (dict, list, tuple, set)


# pylint: disable=too-many-branches, too-many-return-statements
def have_shared_resources(obj1: Any, obj2: Any, visited: set | None = None) -> bool:
    """
    Recursive DFS for checking that two objects do not
    have any (mutable, not callable) shared resources.

    This is crucial for our multiprocessing implementation:
    any shared resources will dramatically increase lock overhead
    and slow down simulations.
    """
    visited = visited or set()

    if id(obj1) in visited or id(obj2) in visited:
        return False

    visited.add(id(obj1))
    visited.add(id(obj2))

    if callable(obj1) and callable(obj2):
        return False

    if not isinstance(obj1, immutable_types) and id(obj1) == id(obj2):
        return True

    # Check for dict, list, set, etc.
    if isinstance(obj1, container_types):
        iter1 = obj1.items() if isinstance(obj1, dict) else enumerate(obj1)
        iter2 = obj2.items() if isinstance(obj2, dict) else enumerate(obj2)
        for (_, item1), (_, item2) in zip(iter1, iter2):
            if item1 is None and item2 is None:
                return False

            if (
                not isinstance(item1, immutable_types)
                and not isinstance(item2, immutable_types)
                and id(item1) == id(item2)
            ):
                return True

            if have_shared_resources(item1, item2, visited):
                return True

    elif (
        not isinstance(obj1, immutable_types)
        and isinstance(obj1, object)
        and hasattr(obj1, "__dict__")
    ):
        for key in vars(obj1).keys():
            attr1 = getattr(obj1, key)
            attr2 = getattr(obj2, key)

            if attr1 is None and attr2 is None:
                return False

            if (
                not isinstance(attr1, immutable_types)
                and not isinstance(attr2, immutable_types)
                and id(attr1) == id(attr2)
            ):
                return True

            if have_shared_resources(attr1, attr2, visited):
                return True

    return False


def test_shared_resources(scenario: Scenario) -> None:
    """
    Test that deepcopying a scenario does not result in
    any shared resources.
    """
    copy1 = deepcopy(scenario)
    copy2 = deepcopy(scenario)
    assert not have_shared_resources(copy1, copy2)
