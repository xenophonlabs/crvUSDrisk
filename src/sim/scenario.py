"""
Provides the `Scenario` class for running simulations.
"""

from typing import List, Tuple, Set, cast, Dict
from itertools import combinations
from datetime import datetime
import pandas as pd
from crvusdsim.pool import get, SimMarketInstance  # type: ignore
from crvusdsim.pool.crvusd.price_oracle.crypto_with_stable_price import Oracle
from crvusdsim.pool.sim_interface import SimController, SimLLAMMAPool
from curvesim.pool.sim_interface import SimCurveCryptoPool
from scipy.stats import gaussian_kde
from .utils import rebind_markets
from ..prices import PriceSample, PricePaths
from ..configs import (
    TOKEN_DTOs,
    get_scenario_config,
    get_price_config,
    get_borrower_kde,
    CRVUSD_DTO,
    ALIASES_LLAMMA,
    MODELLED_MARKETS,
)
from ..configs.tokens import WETH, WSTETH, SFRXETH
from ..modules import ExternalMarket
from ..agents import Arbitrageur, Liquidator, Keeper, Borrower
from ..data_transfer_objects import TokenDTO
from ..types import MarketsType
from ..utils.poolgraph import PoolGraph
from ..logging import get_logger
from ..utils import get_quotes

logger = get_logger(__name__)


class Scenario:
    """
    The `Scenario` object holds ALL of the objects required to run
    a simulation. This includes the pricepaths, the external markets,
    all crvusdsim modules (LLAMMAs, Controllers, etc.), and all agents.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(self, scenario: str, market_names: List[str]):
        for alias in market_names.copy():
            if alias[:2] == "0x":
                alias = ALIASES_LLAMMA[alias]
            assert alias.lower() in MODELLED_MARKETS, ValueError(
                f"Only {MODELLED_MARKETS} markets are supported, not {alias}."
            )
        self.market_names = market_names
        self.config = config = get_scenario_config(scenario)
        self.name: str = config["name"]
        self.description: str = config["description"]
        self.num_steps: int = config["N"]
        self.freq: str = config["freq"]

        self.price_config = get_price_config(
            self.freq, self.config["prices"]["start"], self.config["prices"]["end"]
        )

        self.generate_sim_market()  # must be first
        self.generate_pricepaths()
        self.generate_markets()
        self.generate_agents()

        # FIXME this costs a lot of memory to store
        # but prevents us from having to read it from disk
        # which would cause thread locks.
        self.kde: Dict[str, gaussian_kde] = {}
        for controller in self.controllers:
            pool = controller.AMM
            self.kde[pool.address] = get_borrower_kde(
                pool.address,
                self.config["borrowers"]["start"],
                self.config["borrowers"]["end"],
            )

    def generate_markets(self) -> pd.DataFrame:
        """Generate the external markets for the scenario."""
        start = self.config["quotes"]["start"]
        end = self.config["quotes"]["end"]

        quotes = get_quotes(
            start,
            end,
            self.coins,
        )
        logger.info(
            "Using %d 1Inch quotes from %s to %s",
            quotes.shape[0],
            datetime.fromtimestamp(start),
            datetime.fromtimestamp(end),
        )

        self.quotes_start = start
        self.quotes_end = end

        self.markets: MarketsType = {}
        for pair in self.pairs:
            market = ExternalMarket(pair)
            market.fit(quotes)
            self.markets[pair] = market

        return quotes

    def generate_pricepaths(self) -> None:
        """
        Generate the pricepaths for the scenario.
        """
        self.pricepaths: PricePaths = PricePaths(self.num_steps, self.price_config)

    @property
    def arbed_pools(self) -> list:
        """
        Return the pools that are arbed.
        By default the arbitrageur will arbitrage all
        crvUSD pools against External Markets.

        This includes LLAMMAs and StableSwap pools.
        """
        return self.stableswap_pools + self.llammas + list(self.markets.values())

    def generate_agents(self) -> None:
        """Generate the agents for the scenario."""
        self.arbitrageur: Arbitrageur = Arbitrageur()
        self.liquidator: Liquidator = Liquidator()
        self.keeper: Keeper = Keeper()

        # Set liquidator paths
        self.liquidator.set_paths(self.controllers, self.stableswap_pools, self.markets)

        # Set arbitrage cycles
        self.graph = PoolGraph(self.arbed_pools)
        self.arb_cycle_length = self.config["arb_cycle_length"]
        self.cycles = self.graph.find_cycles(n=self.arb_cycle_length)

        self.borrowers: Dict[str, Borrower] = {}

    def generate_sim_market(self) -> None:
        """
        Generate the crvusd modules to simulate, including
        LLAMMAs, Controllers, StableSwap pools, etc.
        """
        sim_markets: List[SimMarketInstance] = []
        for market_name in self.market_names:
            logger.info("Fetching %s market from subgraph", market_name)

            sim_market = get(
                market_name, bands_data="controller", use_simple_oracle=False
            )

            metadata = sim_market.pool.metadata

            logger.info(
                "Market snapshot as %s",
                datetime.fromtimestamp(int(metadata["timestamp"])),
            )
            logger.info(
                "Bands snapshot as %s",
                datetime.fromtimestamp(int(metadata["bands"][0]["timestamp"])),
            )
            logger.info(
                "Users snapshot as %s",
                datetime.fromtimestamp(int(metadata["userStates"][0]["timestamp"])),
            )

            # huge memory consumption
            del sim_market.pool.metadata["bands"]
            del sim_market.pool.metadata["userStates"]

            sim_markets.append(sim_market)

        # Rebind all markets to use the same shared resources
        rebind_markets(sim_markets)

        # Set scenario attributes
        self.llammas = [sim_market.pool for sim_market in sim_markets]
        self.controllers = [sim_market.controller for sim_market in sim_markets]
        self.oracles = cast(
            List[Oracle], [sim_market.price_oracle for sim_market in sim_markets]
        )
        self.policies = [sim_market.policy for sim_market in sim_markets]
        self.stableswap_pools = sim_markets[0].stableswap_pools
        self.aggregator = sim_markets[0].aggregator
        self.peg_keepers = sim_markets[0].peg_keepers
        self.factory = sim_markets[0].factory
        self.stablecoin = sim_markets[0].stablecoin

        self.tricryptos: Set[SimCurveCryptoPool] = set()
        for sim_market in sim_markets:
            self.tricryptos.update(sim_market.tricrypto)

        self.coins: Set[TokenDTO] = set()
        for pool in self.llammas + self.stableswap_pools:
            for a in pool.coin_addresses:
                if a.lower() != CRVUSD_DTO.address.lower():
                    self.coins.add(TOKEN_DTOs[a.lower()])

        self.pairs: List[Tuple[TokenDTO, TokenDTO]] = [
            (sorted_pair[0], sorted_pair[1])
            for pair in combinations(self.coins, 2)
            for sorted_pair in [sorted(pair)]
        ]

    def update_market_prices(self, sample: PriceSample) -> None:
        """Update market prices with a new sample."""
        for pair in self.pairs:
            self.markets[pair].update_price(sample.prices)

    def prepare_for_run(self, resample_borrowers: bool = True) -> None:
        """
        Prepare all modules for a simulation run.

        Perform initial setup in this order:
        1. Run a single step of price arbitrages.
        2. Liquidate users that were loaded underwater.
        """
        if resample_borrowers:
            self.setup_borrowers()  # randomly sample positions

        sample = self.pricepaths[0]
        ts = int(sample.timestamp.timestamp())

        # Setup timestamp related attrs for all modules
        self._increment_timestamp(ts)

        # Set external market prices
        self.prepare_for_trades(sample)

        # Equilibrate pool prices
        arbitrageur = Arbitrageur()  # diff arbitrageur
        arbitrageur.arbitrage(self.cycles, sample)
        logger.debug(
            "Equilibrated prices with %d arbitrages with total profit %d",
            arbitrageur.count(),
            arbitrageur.profit(),
        )

        # Liquidate users that were loaded underwater
        for controller in self.controllers:
            name = controller.AMM.name.replace("Curve.fi Stablecoin ", "")
            logger.debug("Validating loaded positions in %s Controller.", name)

            to_liquidate = controller.users_to_liquidate()
            n = len(to_liquidate)

            # Check that only a small portion of debt is liquidatable at start
            damage = 0
            for pos in to_liquidate:
                damage += pos.debt
                logger.debug("Liquidating %s: with debt %d.", pos.user, pos.debt)
            pct = round(damage / controller.total_debt() * 100, 2)
            args = (
                "%.2f%% of debt (%d positions) was incorrectly "
                "loaded with <0 health (%d crvUSD) in %s Controller.",
                pct,
                n,
                damage / 1e18,
                name,
            )
            if pct > 1:
                logger.warning(*args)
            else:
                logger.debug(*args)

            controller.after_trades(do_liquidate=True)  # liquidations

    def _increment_timestamp(self, ts: int) -> None:
        """Increment the timestamp for all modules."""
        self.aggregator._increment_timestamp(ts)  # pylint: disable=protected-access

        for llamma in self.llammas:
            llamma._increment_timestamp(ts)  #  pylint: disable=protected-access
            llamma.prev_p_o_time = ts
            llamma.rate_time = ts

        for oracle in self.oracles:
            oracle._increment_timestamp(ts)  #  pylint: disable=protected-access

        for controller in self.controllers:
            controller._increment_timestamp(ts)  #  pylint: disable=protected-access

        for spool in self.stableswap_pools:
            spool._increment_timestamp(ts)  #  pylint: disable=protected-access
            spool.ma_last_time = ts

        for tpool in self.tricryptos:
            tpool._increment_timestamp(ts)  #  pylint: disable=protected-access
            tpool.last_prices_timestamp = ts

        for pk in self.peg_keepers:
            pk._increment_timestamp(ts)  #  pylint: disable=protected-access

    def update_tricrypto_prices(self, sample: PriceSample) -> None:
        """Update TriCrypto prices with a new sample."""
        for tpool in self.tricryptos:
            tpool.prepare_for_run(
                pd.DataFrame(
                    [
                        [
                            # Get pairwise prices for TriCrypto pool assets
                            sample.prices[pair[0].lower()][pair[1].lower()]
                            for pair in list(combinations(tpool.assets.addresses, 2))
                        ]
                    ]
                )
            )

    def update_staked_prices(self, sample: PriceSample) -> None:
        """Update staked prices with a new sample."""
        # Need to know which assets (staked and base)
        for llamma in self.llammas:
            oracle = cast(Oracle, llamma.price_oracle_contract)
            if hasattr(oracle, "staked_oracle"):
                derivative = llamma.COLLATERAL_TOKEN.address
                base = WETH
                assert derivative in [WSTETH, SFRXETH]
                p_staked = int(sample.prices[derivative][base] * 10**18)
                oracle.staked_oracle.update(p_staked)

    def prepare_for_trades(self, sample: PriceSample) -> None:
        """Prepare all modules for a new time step."""
        ts = sample.timestamp
        self.update_market_prices(sample)
        self.update_tricrypto_prices(sample)
        self.update_staked_prices(sample)

        for llamma in self.llammas:
            llamma.prepare_for_trades(ts)  # this increments oracle timestamp

        for controller in self.controllers:
            controller.prepare_for_trades(ts)

        self.aggregator.prepare_for_trades(ts)

        for spool in self.stableswap_pools:
            spool.prepare_for_trades(ts)

        for tpool in self.tricryptos:
            tpool.prepare_for_trades(ts)

        for peg_keeper in self.peg_keepers:
            peg_keeper.prepare_for_trades(ts)

        sample.prices_usd[CRVUSD_DTO.address] = self.aggregator.price() / 1e18

    def after_trades(self) -> None:
        """Perform post processing for all modules at the end of a time step."""

    def perform_actions(self, prices: PriceSample) -> None:
        """Perform all agent actions for a time step."""
        self.arbitrageur.arbitrage(self.cycles, prices)
        self.liquidator.perform_liquidations(self.controllers, prices)
        self.keeper.update(self.peg_keepers)
        # TODO LP

    def setup_borrowers(self) -> None:
        """
        Setup borrowers for the scenario.
        - Shift AMM price up slightly
        - Clear debt positions
        - Resample new debt positions from KDE
        - (AMM price shifts back down in `prepare_for_run`)
        """
        for controller in self.controllers:
            llamma = controller.AMM
            llamma.price_oracle_contract.freeze()
            reset_controller_price(controller)
            clear_controller(controller)
            num_loans = self.config["borrowers"]["num_loans"][
                ALIASES_LLAMMA[llamma.address]
            ]
            for _ in range(num_loans):
                borrower = Borrower()
                success = borrower.create_loan(controller, self.kde[llamma.address])
                if not success:
                    break
                self.borrowers[borrower.address] = borrower
            llamma.price_oracle_contract.unfreeze()
            find_active_band(llamma)
            del self.kde[llamma.address]  # free up memory
            logger.debug(
                "Resampled %d loans with total debt %d in %s.",
                num_loans,
                controller.total_debt(),
                llamma.name,
            )


def find_active_band(llamma: SimLLAMMAPool) -> None:
    """Find the active band for a LLAMMA."""
    min_band = llamma.min_band
    for n in range(llamma.min_band, llamma.max_band):
        if llamma.bands_x[n] == 0 and llamma.bands_y[n] == 0:
            min_band += 1
        if llamma.bands_x[n] == 0 and llamma.bands_y[n] > 0:
            llamma.active_band = n
            break
    llamma.min_band = min_band


def clear_controller(controller: SimController) -> None:
    """Clear all controller states."""
    users = list(controller.loan.keys())
    for user in users:
        debt = controller._debt(user)[0]  # pylint: disable=protected-access
        if debt > 0:
            controller.STABLECOIN._mint(user, debt)  # pylint: disable=protected-access
        controller.repay(debt, user)
        del controller.loan[user]

    assert controller.n_loans == 0
    assert len(controller.loan) == 0
    assert controller.total_debt() == 0


def reset_controller_price(controller: SimController) -> None:
    """
    Reset controller price.
    """
    llamma = controller.AMM

    min_band = llamma.min_band - 3  # Offset by a little bit
    active_band = min_band + 1

    llamma.active_band = active_band
    llamma.min_band = min_band

    p = llamma.p_oracle_up(active_band)
    llamma.price_oracle_contract.last_price = p

    ts = controller._block_timestamp  # pylint: disable=protected-access
    controller.prepare_for_trades(ts + 60 * 60)
    llamma.prepare_for_trades(ts + 60 * 60)
