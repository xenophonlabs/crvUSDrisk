"""
Provides the `Scenario` class for running simulations.
"""

from typing import List, Tuple, Set, cast, Dict
from itertools import combinations
from datetime import datetime
import pandas as pd
from crvusdsim.pool import get, SimMarketInstance  # type: ignore
from crvusdsim.pool.crvusd.price_oracle.crypto_with_stable_price import Oracle
from curvesim.pool.sim_interface import SimCurveCryptoPool
from scipy.stats import gaussian_kde
import numpy as np
from .utils import (
    rebind_markets,
    clear_controller,
    reset_controller_price,
    find_active_band,
)
from ..prices import PriceSample, PricePaths
from ..configs import (
    TOKEN_DTOs,
    get_scenario_config,
    get_price_config,
    get_borrower_kde,
    get_liquidity_config,
    CRVUSD_DTO,
    ALIASES_LLAMMA,
    MODELLED_MARKETS,
)
from ..configs.tokens import WETH, WSTETH, SFRXETH, STABLE_CG_IDS, COINGECKO_IDS_INV
from ..modules import ExternalMarket
from ..agents import Arbitrageur, Liquidator, Keeper, Borrower, LiquidityProvider
from ..data_transfer_objects import TokenDTO
from ..types import MarketsType
from ..utils.poolgraph import PoolGraph
from ..logging import get_logger
from ..utils import get_quotes, get_crvusd_index

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
        self.num_steps: int = config["N"]
        self.freq: str = config["freq"]
        self.subgraph_end_ts = config["subgraph_end_ts"]

        self.price_config = get_price_config(
            self.freq, self.config["prices"]["start"], self.config["prices"]["end"]
        )

        self.target_debt: Dict[
            str, float
        ] | None = None  # must be set by scenario shocks!
        self.target_liquidity_ratio: float | None = (
            None  # must be set by scenario shocks!
        )
        self.apply_shocks()

        self.generate_sim_market()  # must be first
        self.generate_pricepaths()
        self.generate_markets()
        self.generate_agents()

        self.kde: Dict[str, gaussian_kde] = {}
        for controller in self.controllers:
            pool = controller.AMM
            self.kde[pool.address] = get_borrower_kde(
                pool.address,
                config["borrowers"]["start"],
                config["borrowers"]["end"],
            )

        self.liquidity_config = get_liquidity_config(
            config["liquidity"]["start"], config["liquidity"]["end"]
        )

        self.curr_price = self.pricepaths[0]

    @property
    def arbed_pools(self) -> list:
        """
        Return the pools that are arbed.
        By default the arbitrageur will arbitrage all
        crvUSD pools against External Markets.

        This includes LLAMMAs and StableSwap pools.
        """
        return self.stableswap_pools + self.llammas + list(self.markets.values())

    @property
    def total_debt(self) -> int:
        """
        Get total debt in all controllers.
        """
        return sum((controller.total_debt() for controller in self.controllers))

    @property
    def total_crvusd_liquidity(self) -> int:
        """Total crvUSD liquidity in StableSwap Pools."""
        return sum(
            (spool.balances[get_crvusd_index(spool)] for spool in self.stableswap_pools)
        )

    ### ========== Scenario Setup ========== ###

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

    def generate_agents(self) -> None:
        """Generate the agents for the scenario."""
        self.arbitrageur: Arbitrageur = Arbitrageur()
        self.liquidator: Liquidator = Liquidator()
        self.keeper: Keeper = Keeper()

        # Set liquidator paths
        self.liquidator.set_paths(self.controllers, self.stableswap_pools, self.markets)

        # Set arbitrage cycles
        self.graph = PoolGraph(self.arbed_pools)
        self.arb_cycle_length = 3
        self.cycles = self.graph.find_cycles(n=self.arb_cycle_length)

        self.borrowers: Dict[str, Borrower] = {}
        self.lps: Dict[str, LiquidityProvider] = {}

    def generate_sim_market(self) -> None:
        """
        Generate the crvusd modules to simulate, including
        LLAMMAs, Controllers, StableSwap pools, etc.
        """
        sim_markets: List[SimMarketInstance] = []
        for market_name in self.market_names:
            logger.info("Fetching %s market from subgraph", market_name)

            sim_market = get(
                market_name,
                bands_data="controller",
                use_simple_oracle=False,
                end_ts=self.subgraph_end_ts,
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

    def resample_debt(self) -> None:
        """
        Setup borrowers for the scenario.
        - Shift AMM price up slightly
        - Clear debt positions
        - Resample new debt positions from KDE until we hit target debt
        - (AMM price shifts back down in `prepare_for_run`)
        """
        assert self.target_debt, RuntimeError("Target debt not set.")
        for controller in self.controllers:
            llamma = controller.AMM
            llamma.price_oracle_contract.freeze()
            reset_controller_price(controller)
            clear_controller(controller)
            ceiling = self.factory.debt_ceiling[controller.address]
            target_debt = int(
                self.target_debt[ALIASES_LLAMMA[llamma.address]] * ceiling
            )
            while controller.total_debt() < target_debt:
                borrower = Borrower()
                success = borrower.create_loan(controller, self.kde[llamma.address])
                if not success:
                    break
                self.borrowers[borrower.address] = borrower
            leftover = controller.total_debt() - target_debt
            if success and leftover > 0:
                controller.repay(leftover, borrower.address)
            llamma.price_oracle_contract.unfreeze()
            find_active_band(llamma)
            del self.kde[llamma.address]  # free up memory
            logger.debug(
                "Resampled total debt %d in %s.",
                controller.total_debt(),
                llamma.name,
            )

    def resample_liquidity(self) -> None:
        """
        Resample the liquidity in stableswap pools as a
        function of the current total debt.

        Ensure we hit the target liquidity ratio.
        """
        assert self.target_liquidity_ratio, RuntimeError(
            "Target liquidity ratio not set."
        )

        cfg = self.liquidity_config
        total_debt = self.total_debt

        deposit_amounts = []
        total_crvusd_liquidity = 0
        for spool in self.stableswap_pools:
            # Sample liquidity for spool from multivariate normal
            mean = np.array(cfg[spool.address]["mean_vector"])
            cov = np.array(cfg[spool.address]["covariance_matrix"])
            while True:
                _amounts = np.random.multivariate_normal(mean, cov)
                amounts = np.array(
                    [int(b * 1e36 / r) for b, r in zip(_amounts, spool.rates)]
                )
                if all(amount > 0 for amount in amounts):
                    deposit_amounts.append(amounts)
                    total_crvusd_liquidity += amounts[get_crvusd_index(spool)]
                    break

        scale_factor = total_debt / (
            self.target_liquidity_ratio * total_crvusd_liquidity
        )

        for spool, amounts in zip(self.stableswap_pools, deposit_amounts):
            spool.remove_liquidity(spool.totalSupply, [0, 0])
            assert spool.totalSupply == 0
            assert spool.balances == [0, 0]
            # Resample
            lp = LiquidityProvider()
            lp.add_liquidity(
                spool,
                amounts * scale_factor,
            )
            self.lps[lp.address] = lp

        logger.debug(
            "Resampled total crvUSD liquidity %d with ratio %f",
            self.total_crvusd_liquidity,
            self.total_debt / self.total_crvusd_liquidity,
        )

    ### ========== Scenario Shocks ========== ###

    def apply_shocks(self) -> None:
        """Apply the scenario shocks."""
        for shock in self.config["shocks"]:
            if shock["type"] == "mu":
                self.shock_mu(shock)
            elif shock["type"] == "debt":
                self.shock_debt(shock)
            elif shock["type"] == "liquidity":
                self.shock_debt_liquidity_ratio(shock)
            elif shock["type"] == "vol":
                self.shock_vol(shock)

    def shock_mu(self, shock: dict) -> None:
        """
        Shocks the drift for collateral price GBMs.
        """
        for k, v in self.price_config["params"].items():
            if k not in STABLE_CG_IDS:
                token = TOKEN_DTOs[COINGECKO_IDS_INV[k]]
                v["mu"] = shock["target"][token.symbol]

    def shock_debt_liquidity_ratio(self, shock: dict) -> None:
        """
        Shocks the ratio of Debt : crvUSD Liquidity.
        """
        self.target_liquidity_ratio = shock["target"]

    def shock_vol(self, shock: dict) -> None:
        """
        Shocks the volatility for collateral price GBMs.
        """
        for k, v in self.price_config["params"].items():
            if k not in STABLE_CG_IDS:
                token = TOKEN_DTOs[COINGECKO_IDS_INV[k]]
                v["sigma"] = shock["target"][token.symbol]

    def shock_debt(self, shock: dict) -> None:
        """
        Shocks the total amount of debt in the system.
        """
        self.target_debt = shock["target"]

    ### ========== Scenario Execution ========== ###

    def update_market_prices(self, sample: PriceSample) -> None:
        """Update market prices with a new sample."""
        for pair in self.pairs:
            self.markets[pair].update_price(sample.prices)

    def prepare_for_run(self, resample: bool = True) -> None:
        """
        Prepare all modules for a simulation run.

        Perform initial setup in this order:
        1. Run a single step of price arbitrages.
        2. Liquidate users that were loaded underwater.
        """
        if resample:
            self.resample_debt()  # randomly sample positions
            self.resample_liquidity()  # resample crvUSD liquidity

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
                "%.2f%% of debt (%d positions) were incorrectly "
                "loaded with <0 health (%d crvUSD) in %s Controller.",
                pct,
                n,
                damage / 1e18,
                name,
            )
            logger.debug(*args)

            controller.after_trades(do_liquidate=True)  # liquidations

        for llamma in self.llammas:
            llamma.reset_admin_fees()

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

        self.curr_price = sample

    def perform_actions(self, prices: PriceSample) -> None:
        """Perform all agent actions for a time step."""
        assert prices is self.curr_price  # Check we have prepared for trades
        self.arbitrageur.arbitrage(self.cycles, prices)
        self.liquidator.perform_liquidations(self.controllers, prices)
        self.keeper.update(self.peg_keepers)
