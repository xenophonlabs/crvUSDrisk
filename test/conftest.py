"""
Provides the testing configuration.

All network requests are intercepted and mocked. The mocks
expect there to be necessary data in the data/test/ directory.

TODO a lot of tests are "flaky" because they rely on stochastic
debt resamplings or stochastic price updates. It would be *way* 
better to provide static data for these tests to make the tests
deterministic.
"""
from typing import List, Set
import os
import json
from unittest.mock import patch
import pytest
import pandas as pd
from crvusdsim.pool import SimMarketInstance, get_sim_market
from src.configs import LLAMMA_ALIASES, MODELLED_MARKETS
from src.data_transfer_objects import TokenDTO
from src.sim import Scenario

MOCKED_PRICES = {
    "paxos-standard": 0.999346,
    "staked-frax-ether": 2504.84,
    "tbtc": 42937,
    "tether": 0.999809,
    "true-usd": 0.987905,
    "usd-coin": 1.0,
    "weth": 2339.67,
    "wrapped-bitcoin": 42909,
    "wrapped-steth": 2702.52,
}

# Get the testing data
BASE_DIR = os.getcwd()
TEST_DIR = os.path.join(BASE_DIR, "test/data/")
POOL_METADATA_DIR = os.path.join(TEST_DIR, "metadata/")
QUOTES_DIR = os.path.join(TEST_DIR, "quotes/")
assert os.path.exists(TEST_DIR), "Need testing data to run tests."
assert os.path.exists(POOL_METADATA_DIR), "Need pool metadata to run tests."
assert os.path.exists(QUOTES_DIR), "Need quotes to run tests."


@pytest.fixture(autouse=True)
def disable_network_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Raise an error if any network requests are made in testing.
    """

    def stunted_get(*args: list, **kwargs: dict) -> None:
        raise RuntimeError("Network access not allowed during testing!")

    monkeypatch.setattr("requests.get", stunted_get)
    monkeypatch.setattr("requests.post", stunted_get)


def mocked_get_current_prices(coin_ids: List[str]) -> dict:
    """
    Filter mocked prices based on the argument to
    get_current_prices.
    """
    return {k: MOCKED_PRICES[k] for k in coin_ids}


def mocked_get_sim_market(
    market_name: str,
    bands_data: str = "controller",
    use_simple_oracle: bool = False,
    end_ts: int | None = None,  # pylint: disable=unused-argument
) -> SimMarketInstance:
    """
    Returns a SimMarketInstance object using stored pool metadata.
    """
    if "0x" not in market_name:
        market_name = LLAMMA_ALIASES[market_name]

    fn = os.path.join(POOL_METADATA_DIR, f"pool_metadata_{market_name}.json")
    with open(fn, "r", encoding="utf-8") as f:
        pool_metadata = json.load(f)

    bands_x = {
        int(k): int(v) for k, v in pool_metadata["llamma_params"]["bands_x"].items()
    }
    bands_y = {
        int(k): int(v) for k, v in pool_metadata["llamma_params"]["bands_y"].items()
    }

    pool_metadata["llamma_params"]["bands_x"] = bands_x
    pool_metadata["llamma_params"]["bands_y"] = bands_y

    return get_sim_market(
        pool_metadata,
        bands_data=bands_data,
        use_simple_oracle=use_simple_oracle,
    )


def mocked_get_quotes(
    start: int,  # pylint: disable=unused-argument
    end: int,  # pylint: disable=unused-argument
    coins: Set[TokenDTO],  # pylint: disable=unused-argument
) -> pd.DataFrame:
    """
    Returns a dataframe of quotes for the given coins.
    """
    fn = os.path.join(QUOTES_DIR, "quotes.csv")
    return pd.read_csv(fn, index_col=[0, 1])


@pytest.fixture(scope="session")
def scenario() -> Scenario:
    """
    Returns a scenario for testing, while mocking all network requests.

    Note
    ----
    If changes are made to the crvUSDsim SimMarketInstance class, the
    metadata in data/test/metadata/ will need to be updated.
    """
    with (
        patch("src.configs.get_current_prices", side_effect=mocked_get_current_prices),
        patch("src.sim.scenario.get", side_effect=mocked_get_sim_market),
        patch("src.sim.scenario.get_quotes", side_effect=mocked_get_quotes),
    ):
        _scenario = Scenario("baseline", MODELLED_MARKETS)
        _scenario.prepare_for_run(resample=False)
        return _scenario
