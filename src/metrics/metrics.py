"""Provides classes for each metric."""
from typing import List, Dict, cast
from collections import defaultdict
import numpy as np
from .base import Metric
from .utils import entity_str
from ..utils import get_crvusd_index
from ..configs.tokens import CRVUSD
from ..sim.scenario import Scenario


class BadDebtMetric(Metric):
    """
    Calculates the bad debt in a LLAMMA/Controller
    at  the current timestep.
    """

    key_metric = "Bad Debt Pct"

    def _config(self) -> Dict[str, List[str]]:
        cfg = {"Bad Debt Pct": ["max"]}
        for controller in self.scenario.controllers:
            cfg[f"Bad Debt Pct on {entity_str(controller, 'controller')}"] = ["max"]
        return cfg

    def compute(self, **kwargs: dict) -> Dict[str, float]:
        """Compute bad debt."""
        total = 0
        val = {}
        for controller in self.scenario.controllers:
            healths = kwargs["healths"][controller.address]
            debts = kwargs["debts"][controller.address]
            bad_debt = debts[np.where(healths < 0)].sum() / 1e18
            val[f"Bad Debt Pct on {entity_str(controller, 'controller')}"] = (
                bad_debt / kwargs["initial_debts"][controller.address] * 100
            )
            total += bad_debt
        total_debt = cast(float, kwargs["total_initial_debt"])
        val["Bad Debt Pct"] = total / total_debt * 100
        return val


class SystemHealthMetric(Metric):
    """
    Calculates the weighted average
    system health in a LLAMMA/Controller at the
    current timestep.
    """

    key_metric = "System Health"

    def _config(self) -> Dict[str, List[str]]:
        cfg = {"System Health": ["min"]}
        for controller in self.scenario.controllers:
            cfg[f"System Health on {entity_str(controller, 'controller')}"] = [
                "min",
            ]
        return cfg

    def compute(self, **kwargs: dict) -> Dict[str, float]:
        """Compute system health."""
        val = {}
        all_healths = np.array([])
        all_debts = np.array([])
        for controller in self.scenario.controllers:
            healths = kwargs["healths"][controller.address]
            debts = kwargs["debts"][controller.address]
            all_healths = np.append(all_healths, healths)
            all_debts = np.append(all_debts, debts)
            val[f"System Health on {entity_str(controller, 'controller')}"] = (
                (healths * debts).sum() / debts.sum() / 1e18
            )
        val["System Health"] = (all_healths * all_debts).sum() / all_debts.sum() / 1e18
        return val


class BorrowerLossMetric(Metric):
    """
    Calculates losses to borrowers from hard and soft
    liquidations.

    From hard liquidations: loss = liquidator profit = $collateral liquidated - $debt repaid
    Example:
        10 ETH liquidated and 20,000 crvUSD repaid.
        $ETH = $2,000 (sim price) and $crvUSD = $0.99 (aggregator price).
        Liquidator Profit = Borrower Loss = $200 USD.
    From soft liquidations: loss = arbitrageur profit = $tokens bought - $tokens sold
    Example:
        1 ETH bought, 2,000 crvUSD sold (by arbitrageur).
        $ETH = $2,000 (sim price) and $crvUSD = $0.99 (aggregator price).
        Arbitrageur Profit = Borrower Loss = $20 USD.

    Refer to :class:`src.agents.agent.Agent.update_borrower_losses`.

    Note
    ----
    $ indicates MTM USD value at current *market* prices (use aggregator price for crvUSD).
    """

    key_metric = "Borrower Loss Pct"

    def _config(self) -> Dict[str, List[str]]:
        return {
            "Borrower Loss Pct": ["max"],  # == last
            "Hard Liquidation Loss Pct": ["max"],  # == last
            "Soft Liquidation Loss Pct": ["max"],  # == last
        }

    def compute(self, **kwargs: dict) -> Dict[str, float]:
        """Compute borrower loss."""
        total_debt = cast(float, kwargs["total_initial_debt"])
        hard_loss = self.scenario.liquidator.borrower_loss
        soft_loss = self.scenario.arbitrageur.borrower_loss
        borrower_loss = hard_loss + soft_loss
        return {
            "Borrower Loss Pct": borrower_loss / total_debt * 100,
            "Hard Liquidation Loss Pct": hard_loss / total_debt * 100,
            "Soft Liquidation Loss Pct": soft_loss / total_debt * 100,
        }


class ValueLeakageMetric(Metric):
    """
    Calculates the value leaked to the system's maintainers,
    including Arbitrageurs, Liquidators, and Keepers.

    This is a superset of the BorrowerLoss metric, which only
    accounts for Liquidator profits, some Arbitrageur profits, and
    no Keeper profits.
    """

    key_metric = "Value Leakage"

    def _config(self) -> Dict[str, List[str]]:
        cfg = {"Value Leakage": ["max"]}

        cfg["Keeper Profit"] = ["max"]
        cfg["Keeper Count"] = ["max"]
        for pk in self.scenario.peg_keepers:
            _pk = entity_str(pk, "pk")
            cfg[f"Keeper Profit on {_pk}"] = ["max"]
            cfg[f"Keeper Count on {_pk}"] = ["max"]

        cfg["Liquidator Profit"] = ["max"]
        cfg["Liquidator Count"] = ["max"]
        for controller in self.scenario.controllers:
            _controller = entity_str(controller, "controller")
            cfg[f"Liquidator Profit on {_controller}"] = ["max"]
            cfg[f"Liquidator Count on {_controller}"] = ["max"]

        cfg["Arbitrageur Profit"] = ["max"]
        cfg["Arbitrageur Count"] = ["max"]

        return cfg

    def compute(self, **kwargs: dict) -> Dict[str, float]:
        val = {}

        keeper = self.scenario.keeper
        val["Keeper Profit"] = keeper.profit()
        val["Keeper Count"] = keeper.count()
        for pk in self.scenario.peg_keepers:
            _pk = entity_str(pk, "pk")
            val[f"Keeper Profit on {_pk}"] = keeper.profit(pk.address)
            val[f"Keeper Count on {_pk}"] = keeper.count(pk.address)

        liquidator = self.scenario.liquidator
        val["Liquidator Profit"] = liquidator.profit()
        val["Liquidator Count"] = liquidator.count()
        for controller in self.scenario.controllers:
            _controller = entity_str(controller, "controller")
            val[f"Liquidator Profit on {_controller}"] = liquidator.profit(
                controller.address
            )
            val[f"Liquidator Count on {_controller}"] = liquidator.count(
                controller.address
            )

        val["Arbitrageur Profit"] = self.scenario.arbitrageur.profit()
        val["Arbitrageur Count"] = self.scenario.arbitrageur.count()

        val["Value Leakage"] = (
            val["Liquidator Profit"] + val["Arbitrageur Profit"] + val["Keeper Profit"]
        )

        return val


class PegStrengthMetric(Metric):
    """
    Calculates the strength of the crvUSD peg.
    """

    key_metric = "Aggregator Price"

    def _config(self) -> Dict[str, List[str]]:
        cfg = {
            "Aggregator Price": ["min", "max"],
        }
        for spool in self.scenario.stableswap_pools:
            cfg[f"{entity_str(spool, 'stableswap')} Price"] = ["min", "max"]
        return cfg

    def compute(self, **kwargs: dict) -> Dict[str, float]:
        """Compute peg strength."""
        val = {"Aggregator Price": self.scenario.aggregator.price() / 1e18}
        for spool in self.scenario.stableswap_pools:
            i = get_crvusd_index(spool)
            val[f"{entity_str(spool, 'stableswap')} Price"] = (
                spool.price(i ^ 1, i) / 1e18
            )
        return val


class LiquidationsMetric(Metric):
    """
    Calculates the hard liquidation volume.
    """

    key_metric = "Debt Liquidated Pct"

    def _config(self) -> Dict[str, List[str]]:
        cfg = {
            "Debt Liquidated Pct": ["max"],
        }

        for controller in self.scenario.controllers:
            cfg[f"Debt Liquidated Pct on {entity_str(controller, 'controller')}"] = [
                "max"
            ]

        return cfg

    def compute(self, **kwargs: dict) -> Dict[str, float]:
        """Compute liquidation metrics."""
        val = {}
        liquidator = self.scenario.liquidator
        total_debt_liquidated = 0.0
        for controller in self.scenario.controllers:
            _controller = entity_str(controller, "controller")
            debt_liquidated = liquidator.debt_repaid[controller.address]
            val[f"Debt Liquidated Pct on {_controller}"] = (
                debt_liquidated / kwargs["initial_debts"][controller.address] * 100
            )
            total_debt_liquidated += debt_liquidated

        total_debt = cast(float, kwargs["total_initial_debt"])
        val["Debt Liquidated Pct"] = total_debt_liquidated / total_debt * 100

        return val


class PegKeeperMetric(Metric):
    """
    Calculates the total PK debt.
    """

    key_metric = "PK Debt"

    def _config(self) -> Dict[str, List[str]]:
        cfg = {
            "PK Debt": ["max"],
        }
        for pk in self.scenario.peg_keepers:
            cfg[f"{entity_str(pk, 'pk')} Debt"] = ["max"]
        return cfg

    def compute(self, **kwargs: dict) -> Dict[str, float]:
        """Compute PK debt."""
        val = {}
        for pk in self.scenario.peg_keepers:
            val[f"{entity_str(pk, 'pk')} Debt"] = pk.debt / 1e18
        val["PK Debt"] = sum(val.values())
        return val


class LiquidityMetric(Metric):
    """
    Tracks total crvUSD liquidity in stableswap pools.
    """

    key_metric = "Total crvUSD Liquidity"

    def _config(self) -> Dict[str, List[str]]:
        cfg = {"Total crvUSD Liquidity": ["min"]}

        for spool in self.scenario.stableswap_pools:
            cfg[f"{entity_str(spool, 'stableswap')} crvUSD Liquidity"] = ["min"]

        return cfg

    def compute(self, **kwargs: dict) -> Dict[str, float]:
        """Compute crvUSD liquidity."""
        val = {}

        total_liquidity = 0
        for spool in self.scenario.stableswap_pools:
            liquidity = spool.balances[get_crvusd_index(spool)] / 1e18
            val[f"{entity_str(spool, 'stableswap')} crvUSD Liquidity"] = liquidity
            total_liquidity += liquidity

        val["Total crvUSD Liquidity"] = total_liquidity

        return val


class DebtMetric(Metric):
    """
    Track total debt in each Controller.
    """

    key_metric = "Total Debt"

    def _config(self) -> Dict[str, List[str]]:
        cfg = {"Total Debt": ["mean"]}

        for controller in self.scenario.controllers:
            cfg[f"{entity_str(controller, 'controller')} Total Debt"] = ["mean"]

        return cfg

    def compute(self, **kwargs: dict) -> Dict[str, float]:
        """Compute miscellaneous metrics."""
        val = {}

        total_debt = 0
        for controller in self.scenario.controllers:
            debt = kwargs["debts"][controller.address].sum() / 1e18
            val[f"{entity_str(controller, 'controller')} Total Debt"] = debt
            total_debt += debt

        val["Total Debt"] = total_debt

        return val


class PriceMetric(Metric):
    """
    Track LLAMMA Oracle prices vs market prices.
    """

    key_metric = "Worst Oracle Error Pct"

    def _config(self) -> Dict[str, List[str]]:
        cfg = {"Worst Oracle Error Pct": ["max"]}

        for llamma in self.scenario.llammas:
            cfg[f"{entity_str(llamma, 'llamma')} Oracle Price"] = ["mean"]
            cfg[f"{entity_str(llamma, 'llamma')} Oracle Error Pct"] = ["max"]

        for tpool in self.scenario.tricryptos:
            coins = tpool.coin_names
            col = f"{entity_str(tpool, 'tricrypto')} Oracle Price"
            cfg[" ".join([col, coins[1]])] = ["mean"]
            cfg[" ".join([col, coins[2]])] = ["mean"]

        return cfg

    def compute(self, **kwargs: dict) -> Dict[str, float]:
        """Compute price errors."""
        val = {}

        errors = []
        for llamma in self.scenario.llammas:
            oracle_price = llamma.price_oracle() / 1e18
            market_price = self.scenario.curr_price.prices_usd[
                llamma.COLLATERAL_TOKEN.address
            ]
            oracle_error_pct = abs(market_price - oracle_price) / market_price * 100
            errors.append(oracle_error_pct)
            val[f"{entity_str(llamma, 'llamma')} Oracle Price"] = oracle_price
            val[f"{entity_str(llamma, 'llamma')} Oracle Error Pct"] = oracle_error_pct

        for tpool in self.scenario.tricryptos:
            coins = tpool.coin_names
            col = f"{entity_str(tpool, 'tricrypto')} Oracle Price"
            oracle_prices = tpool.price_oracle()
            val[" ".join([col, coins[1]])] = oracle_prices[0] / 1e18
            val[" ".join([col, coins[2]])] = oracle_prices[1] / 1e18

        val["Worst Oracle Error Pct"] = max(errors)

        return val


class ProfitsMetric(Metric):
    """
    Compute the profits to LLAMMA from swaps as a
    percentage of total debt.
    """

    key_metric = "Net LLAMMA Profit Pct"

    def __init__(self, scenario: Scenario):
        super().__init__(scenario)
        # Track cumulative fees
        self.bands_fees_x: Dict[str, int] = defaultdict(int)
        self.bands_fees_y: Dict[str, int] = defaultdict(int)
        self.profit: Dict[str, float] = defaultdict(float)

    def _config(self) -> Dict[str, List[str]]:
        cfg = {"Net LLAMMA Profit Pct": ["max"]}

        for llamma in self.scenario.llammas:
            cfg[f"LLAMMA Profit on {entity_str(llamma, 'llamma')} Pct"] = ["max"]

        return cfg

    def compute(self, **kwargs: dict) -> Dict[str, float]:
        val = {}

        prices = self.scenario.curr_price.prices_usd
        crvusd_price = prices[CRVUSD]

        profit = 0.0
        for controller in self.scenario.controllers:
            llamma = controller.AMM

            # Note that all fees accrue to bands_fees_x because
            # admin fees are 0.
            llamma_bands_fees_x = sum(llamma.bands_fees_x.values())
            llamma_bands_fees_y = sum(llamma.bands_fees_y.values())

            bands_fees_x = llamma_bands_fees_x - self.bands_fees_x[llamma.address]
            bands_fees_y = llamma_bands_fees_y - self.bands_fees_y[llamma.address]

            profit_x = bands_fees_x * crvusd_price / 1e18
            profit_y = bands_fees_y * prices[llamma.COLLATERAL_TOKEN.address] / 1e18
            profit_llamma = self.profit[llamma.address] + profit_x + profit_y  # cum

            val[f"LLAMMA Profit on {entity_str(llamma, 'llamma')} Pct"] = (
                profit_llamma / kwargs["initial_debts"][controller.address] * 100
            )
            profit += profit_llamma

            # Update
            self.bands_fees_x[llamma.address] = llamma_bands_fees_x
            self.bands_fees_y[llamma.address] = llamma_bands_fees_y
            self.profit[llamma.address] = profit_llamma

        val["Net LLAMMA Profit Pct"] = (
            profit / cast(float, kwargs["total_initial_debt"]) * 100
        )

        return val
