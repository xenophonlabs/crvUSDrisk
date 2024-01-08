"""Provides classes for each metric."""
import numpy as np
from .base import Metric
from .utils import entity_str
from ..utils import get_crvusd_index


class BadDebtMetric(Metric):
    """
    Calculates the bad debt in a LLAMMA/Controller
    at  the current timestep.

    TODO extend to multiple LLAMMAs & Controllers?
    """

    key_metric = "Bad Debt"

    def _config(self):
        return {"Bad Debt": ["mean", "max"]}

    def compute(self, **kwargs):
        """Compute bad debt."""
        healths = kwargs.get("healths")
        debts = kwargs.get("debts")
        return {"Bad Debt": debts[np.where(healths < 0)].sum() / 1e18}


class SystemHealthMetric(Metric):
    """
    Calculates the weighted average
    system health in a LLAMMA/Controller at the
    current timestep.

    TODO extend to multiple LLAMMAs & Controllers?
    """

    key_metric = "System Health"

    def _config(self):
        return {"System Health": ["mean", "min"]}

    def compute(self, **kwargs):
        """Compute system health."""
        healths = kwargs.get("healths")
        debts = kwargs.get("debts")
        return {"System Health": (healths * debts).sum() / debts.sum() / 1e18}


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

    key_metric = "Borrower Loss"

    def _config(self):
        return {
            "Borrower Loss": ["max"],  # == last
            "Hard Liquidation Losses": ["max"],  # == last
            "Soft Liquidation Losses": ["max"],  # == last
        }

    def compute(self, **kwargs):
        """Compute borrower loss."""
        hard_loss = self.scenario.liquidator.borrower_loss
        soft_loss = self.scenario.arbitrageur.borrower_loss
        borrower_loss = hard_loss + soft_loss
        return {
            "Borrower Loss": borrower_loss,
            "Hard Liquidation Losses": hard_loss,
            "Soft Liquidation Losses": soft_loss,
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

    def _config(self):
        return {
            "Value Leakage": ["max"],
            "Keeper Profit": ["max"],  # Using virtual price
            "Keeper Count": ["max"],
            "Liquidator Profit": ["max"],
            "Liquidator Count": ["max"],
            "Arbitrageur Profit": ["max"],  # == last
            "Arbitrageur Count": ["max"],
        }

    def compute(self, **kwargs):
        liquidator_profit = self.scenario.liquidator.profit
        arbitrageur_profit = self.scenario.arbitrageur.profit
        keeper_profit = self.scenario.keeper.profit
        value_leakage = liquidator_profit + arbitrageur_profit + keeper_profit
        return {
            "Value Leakage": value_leakage,
            "Keeper Profit": keeper_profit,
            "Keeper Count": self.scenario.keeper.count,
            "Liquidator Profit": liquidator_profit,
            "Liquidator Count": self.scenario.liquidator.count,
            "Arbitrageur Profit": arbitrageur_profit,
            "Arbitrageur Count": self.scenario.arbitrageur.count,
        }


class PegStrengthMetric(Metric):
    """
    Calculates the strength of the crvUSD peg.
    """

    key_metric = "Peg Strength"

    def _config(self):
        cfg = {
            "Peg Strength": ["mean", "min", "max"],
        }
        for spool in self.scenario.stableswap_pools:
            cfg[f"{entity_str(spool, 'stableswap')} Price"] = ["mean", "min", "max"]
        return cfg

    def compute(self, **kwargs):
        """Compute peg strength."""
        val = {"Peg Strength": self.scenario.aggregator.price() / 1e18}
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

    key_metric = "Collateral Liquidated"

    def _config(self):
        return {
            "Collateral Liquidated": ["max"],
            "Debt Repaid": ["max"],
            "Liquidation Count": ["max"],
        }

    def compute(self, **kwargs):
        """Compute liquidation metrics."""
        return {
            "Collateral Liquidated": self.scenario.liquidator.collateral_liquidated,
            "Debt Repaid": self.scenario.liquidator.debt_repaid,
            "Liquidation Count": self.scenario.liquidator.count,
        }


class PegKeeperMetric(Metric):
    """
    Calculates the total PK debt.
    """

    key_metric = "PK Debt"

    def _config(self):
        cfg = {
            "PK Debt": ["mean", "max"],
        }
        for pk in self.scenario.peg_keepers:
            cfg[f"{entity_str(pk, 'pk')} Debt"] = ["mean", "max"]
        return cfg

    def compute(self, **kwargs):
        """Compute PK debt."""
        val = {}
        for pk in self.scenario.peg_keepers:
            val[f"{entity_str(pk, 'pk')} Debt"] = pk.debt / 1e18
        val["PK Debt"] = sum(val.values())
        return val


class MiscMetric(Metric):
    """
    Miscellaneous metrics that are useful to look at and sanity check.
    """

    key_metric = "crvUSD Total Supply"

    def _config(self):
        return {
            "crvUSD Total Supply": ["max"],
            f"{entity_str(self.scenario.controller, 'controller')} Total Debt": [
                "mean"
            ],
            f"{entity_str(self.scenario.llamma, 'llamma')} Price": ["mean"],
            f"{entity_str(self.scenario.llamma, 'llamma')} Oracle Price": ["mean"],
        }

    def compute(self, **kwargs):
        """Compute miscellaneous metrics."""
        llamma = self.scenario.llamma
        controller = self.scenario.controller
        debts = kwargs.get("debts")
        return {
            "crvUSD Total Supply": self.scenario.stablecoin.totalSupply / 1e18,
            f"{entity_str(controller, 'controller')} Total Debt": debts.sum() / 1e18,
            f"{entity_str(llamma, 'llamma')} Price": llamma.get_p() / 1e18,
            f"{entity_str(llamma, 'llamma')} Oracle Price": llamma.price_oracle()
            / 1e18,
        }
