"""
Provides the `MetricsResult` and `MetricsProcessor` classes,
as well as some helper functions.

TODO consider converting to curvesim metrics architecture.
If we do so, the `MetricsProcessor` is really the `StateLog`.
"""
from typing import Any, List
import pandas as pd
import numpy as np
from crvusdsim.pool.sim_interface import SimController
from ..sim.scenario import Scenario
from ..utils import get_crvusd_index


def entity_str(entity: Any, type_: str):
    """
    Get a simplified name for the pool
    to use in metrics column names.
    """
    if type_ == "aggregator":
        return "aggregator"
    if type_ == "stablecoin":
        return "stablecoin"

    if type_ == "llamma":
        name = entity.name.replace("Curve.fi Stablecoin ", "")
    elif type_ == "controller":
        name = entity.AMM.name.replace("Curve.fi Stablecoin ", "")
    elif type_ == "stableswap":
        name = entity.name.replace("Curve.fi Factory Plain Pool: ", "")
        name = name.replace("/", "_")
    elif type_ == "pk":
        name = entity.POOL.name.replace("Curve.fi Stablecoin ", "")
    else:
        raise ValueError("Invalid type_.")

    return "_".join([type_, name])


class MetricsResult:  # pylint: disable=too-few-public-methods
    """
    Stores metrics data for a single simulation
    """

    def __init__(self, df: pd.DataFrame):
        pass


class MetricsProcessor:
    """
    Stores and massages metrics data
    for a single simulation

    Metrics
    -------
    Arbitrageur: Profit
    Arbitrageur: Count
    Liquidator:  Profit
    Liquidator:  Count
    Keeper: Profit <- when are liquidations "bad"? TODO
    Keeper: Count
    LLAMMA: Arbitrage volume TODO
    LLAMMA: Price
    LLAMMA: Oracle price
    LLAMMA: Fees x
    LLAMMA: Fees y
    LLAMMA: Balances FIXME currently just sum over bands
    Controller: System Health
    Controller: Bad Debt
    Controller: Number of loans
    Controller: Debt
    Controller: Liquidation volume TODO
    Controller: Users to liquidate TODO
    StableSwap: Price
    StableSwap: MA Price
    StableSwap: LP Token Supply
    StableSwap: Virtual Price
    StableSwap: Balances
    PK: Debt
    PK: Profit
    PK: Minted TODO
    PK: Burned TODO
    Aggregator: Price
    ERC20: Total Supply
    ERC20: Volume TODO
    ERC20: Net unbacked crvusd TODO
    """

    columns_base = [
        "timestamp",
    ]

    # TODO could have one liquidator for each LLAMMA
    # TODO could have one keeper for each PK
    columns_agents = [
        "arbitrageur_profit",
        "arbitrageur_count",
        "keeper_profit",
        "keeper_count",
        "liquidator_profit",
        "liquidator_count",
    ]

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

        columns_sim = []

        llamma_str = entity_str(scenario.llamma, "llamma")
        columns_sim.extend(
            [
                llamma_str + "_price",
                llamma_str + "_oracle_price",
                llamma_str + "_fees_x",
                llamma_str + "_fees_y",
                llamma_str + "_bal_x",
                llamma_str + "_bal_y",
            ]
        )

        controller_str = entity_str(scenario.controller, "controller")
        columns_sim.extend(
            [
                controller_str + "_system_health",
                controller_str + "_bad_debt",
                controller_str + "_num_loans",
                controller_str + "_debt",
                # controller_str + "_liquidation_volume",
            ]
        )

        for spool in scenario.stableswap_pools:
            spool_str = entity_str(spool, "stableswap")
            columns_sim.extend(
                [
                    spool_str + "_price",
                    spool_str + "_ma_price",
                    spool_str + "_lp_supply",
                    spool_str + "_virtual_price",
                ]
            )
            for symbol in spool.assets.symbols:
                columns_sim.append("_".join([spool_str, symbol, "bal"]))

        for pk in scenario.peg_keepers:
            pk_str = entity_str(pk, "pk")
            columns_sim.extend(
                [
                    pk_str + "_debt",
                    pk_str + "_profit",
                    # pk_str + "_minted",
                    # pk_str + "_burned",
                ]
            )

        aggregator_str = entity_str(scenario.aggregator, "aggregator")
        columns_sim.append(aggregator_str + "_price")

        stablecoin_str = entity_str(scenario.stablecoin, "stablecoin")
        columns_sim.append(
            stablecoin_str + "_total_supply",
        )

        self.columns_sim = columns_sim
        self.results: List[dict] = []
        self.columns = self.columns_base + self.columns_agents + self.columns_sim

        # Initial state
        self.initial_state = self.update(inplace=False)

    def update(self, inplace: bool = True) -> dict:
        """
        Collect metrics for the current timestep of the sim.
        """
        scenario = self.scenario
        res: List[Any] = [scenario.timestamp]

        # Agents
        agents = [
            scenario.arbitrageur,
            scenario.keeper,
            scenario.liquidator,
        ]
        for agent in agents:
            res.append(agent.profit)
            res.append(agent.count)

        # LLAMMA
        res.append(scenario.llamma.price(0, 1))
        res.append(scenario.llamma.price_oracle())
        res.append(scenario.llamma.admin_fees_x)
        res.append(scenario.llamma.admin_fees_y)
        res.append(sum(scenario.llamma.bands_x.values()))
        res.append(sum(scenario.llamma.bands_y.values()))

        # Controller
        res.append(controller_system_health(scenario.controller))
        res.append(controller_bad_debt(scenario.controller))
        res.append(scenario.controller.n_loans)
        res.append(scenario.controller.total_debt())

        # StableSwap
        for spool in scenario.stableswap_pools:
            i = get_crvusd_index(spool)
            res.append(spool.price(i ^ 1, i))
            res.append(spool.price_oracle())
            res.append(spool.totalSupply)
            res.append(spool.get_virtual_price())
            for i in range(len(spool.assets.symbols)):
                res.append(spool.balances[i])

        # PegKeeper
        for pk in scenario.peg_keepers:
            res.append(pk.debt)
            res.append(pk.calc_profit())

        # Aggregator
        res.append(scenario.aggregator.price())

        # Stablecoin
        res.append(scenario.stablecoin.totalSupply)

        d = dict(zip(self.columns, res))
        if inplace:
            self.results.append(d)

        return d

    @property
    def df(self) -> pd.DataFrame:
        """Parse metrics results into df."""
        return pd.DataFrame(self.results)

    def process(self) -> MetricsResult:
        """Process timeseries df into metrics result."""
        return MetricsResult(self.df)


def controller_system_health(controller: SimController) -> int:
    """
    Calculate the system health of a controller.
    We calculate this as a weighted average of user
    health, where weights are each user's initial debt.
    TODO use current debt instead of initial debt
    """
    return (
        np.array(
            [
                controller.health(user, full=True) * loan.initial_debt
                for user, loan in controller.loan.items()
            ]
        ).sum()
        / controller.total_debt()
    )


def controller_bad_debt(controller: SimController) -> int:
    """
    Calculate net bad debt in controller.
    We define bad debt as the debt of users with
    health < 0.
    TODO use current debt instead of initial debt
    """
    bad_debt = 0
    for user, loan in controller.loan.items():
        if controller.health(user, full=True) < 0:
            bad_debt += loan.initial_debt
    return bad_debt
