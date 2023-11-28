from typing import List
from scipy.optimize import root_scalar
import logging
import math
from .agent import Agent
from ..modules import ExternalMarket
from ..types.cycle import Swap, Liquidation, Cycle
from crvusdsim.pool.crvusd.controller import Position
from crvusdsim.pool.sim_interface import SimController
from crvusdsim.pool.sim_interface.sim_stableswap import SimCurveStableSwapPool
from ..utils import get_crvUSD_index
from dataclasses import dataclass


@dataclass
class Path:
    basis_token: str  # TODO use Token class or something
    crvusd_pool: SimCurveStableSwapPool
    collat_pool: ExternalMarket


class Liquidator(Agent):
    """
    Liquidator performs hard liquidations on LLAMMAs.
    """

    basis_tokens: list = [
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
    ]

    paths: List[Path] = []

    def __init__(self, tolerance: float = 0):
        assert tolerance >= 0
        self.tolerance = tolerance
        self._profit = 0
        self._count = 0

    def set_paths(
        self,
        controller: SimController,
        crvUSD_pools: List[SimCurveStableSwapPool],
        collat_pools: List[ExternalMarket],
    ):
        """
        Set the paths for liquidations. Currently:
        1. Purchase crvUSD from basis_token/crvUSD pools.
        2. Liquidate the user in the Controller.
        3. Sell collateral for basis_token in collateral/basis_token
        External markets.
        """
        self.paths = []
        for basis_token in self.basis_tokens:
            # Get basis_token/crvUSD pool
            for pool in crvUSD_pools:
                coins = [c.address for c in pool.coins]
                if basis_token in coins:
                    crvusd_pool = pool
                    # FIXME assuming only one pool for each basis token
                    break

            # Get collateral/basis_token pool
            collat_pool = collat_pools[controller.COLLATERAL_TOKEN.address][basis_token]
            self.paths.append(Path(basis_token, crvusd_pool, collat_pool))

    def perform_liquidations(
        self,
        controller: SimController,
    ) -> List[float]:
        """
        Loops through all liquidatable users and liquidates if profitable.

        Parameters
        ----------
        controller : Controller
            Controller object.

        Returns
        -------
        total_profit : float
            Profit in USDC units from Liquidations.
        underwater_debt : float
            Total crvUSD debt of underwater positions.

        TODO liquidations should be ordered by profitability
        """
        to_liquidate = controller.users_to_liquidate()

        logging.info(f"There are {len(to_liquidate)} users to liquidate.")

        if len(to_liquidate) == 0:
            return 0, 0

        underwater_debt = 0
        total_profit = 0

        for position in to_liquidate:
            profit = self.maybe_liquidate(position, controller)

            # TODO move this to maybe_liquidate
            if profit > self.tolerance:
                total_profit += profit
                self._count += 1
            else:
                # Liquidation was missed
                underwater_debt += position.debt

        self._profit += total_profit

        return total_profit, underwater_debt

    def maybe_liquidate(
        self,
        position: Position,
        controller: SimController,
    ) -> float:
        """
        This is the hard liquidation:
        1. Liquidator checks the crvUSD debt they'll have to repay.
        2. For each basis token (e.g. USDC, USDT) they:
            a. Compute how much of the basis token they must swap
            to obtain the necessary crvUSD.
            b. Compute how much of the basis token they receive
            from selling the corresponding collateral.
            c. Profit = b - a
        3. They take the most profitable route. If this profit > 0,
        they perform the liquidations.
        or USDT) that gives them the most profit, if this profit > 0.

        Parameters
        ----------
        controller : Controller
            Controller object
        position : Position
            Position object to liquidate

        Returns
        -------
        float
            profit in basis token units

        Note
        ----
        TODO incorporate liquidations that source crvUSD partly from
        USDC and partly from USDT.
        TODO incorporate liquidations against other tokens (not just
        USDC and USDT).
        TODO use the ERC20 dataclass for all token objects in codebase
        """
        user = position.user
        health = position.health

        # TODO int casting should occur in controller
        to_repay = int(controller.tokens_to_liquidate(user))
        # TODO if to_repay == 0: perform liquidation.
        _, y = controller.AMM.get_sum_xy(user)
        y = int(y)

        best = None
        best_expected_profit = -math.inf

        assert self.paths, "Liquidator paths not set."
        for path in self.paths:
            crvusd_pool = path.crvusd_pool
            collat_pool = path.collat_pool

            # TODO Abstract this into the `Cycle.optimize` function

            # basis token -> crvUSD
            j = get_crvUSD_index(crvusd_pool)
            i = j ^ 1
            amt_in = self.search(crvusd_pool, i, j, to_repay)
            trade1 = Swap(crvusd_pool, i, j, amt_in)

            # crvUSD -> collateral
            trade2 = Liquidation(controller, position, to_repay)

            # collateral -> basis token
            amt_out = collat_pool.trade(0, 1, y / 1e18)  # TODO decimals
            trade3 = Swap(collat_pool, 0, 1, y)

            # TODO the decimals are completely wrong
            amt_in_ = trade1.amt / 10 ** trade1.get_decimals(i)
            expected_profit = amt_out - amt_in_

            cycle = Cycle([trade1, trade2, trade3], expected_profit=expected_profit)

            if not best or expected_profit > best_expected_profit:
                best = cycle
                best_expected_profit = expected_profit

        if best_expected_profit > self.tolerance:
            logging.info(
                f"Liquidating user {user} with expected profit: {best_expected_profit}."
            )
            profit = best.execute()
            logging.info(f"Liquidated user {user} with profit: {profit}.")
            return profit
        else:
            logging.info(
                f"Missed liquidation for user {user}. Health: {health}. Expected profit: {best_expected_profit}."
            )
            return 0

    def search(self, pool: SimCurveStableSwapPool, i: int, j: int, amt_out: int) -> int:
        """
        Find the amt_in required to get the desired
        amt_out from a swap.

        Currently only meant for USDC or USDT ->
        crvUSD.

        TODO move this to SimCurveStableSwapPool
        """

        amt_out = float(amt_out)  # For scipy

        assert isinstance(pool, SimCurveStableSwapPool)

        def loss(amt_in: int, pool: SimCurveStableSwapPool, i: int, j: int):
            """
            Loss function for optimization. Very simple:
            we just want to minimize the diff between the
            desired amt_out, and the actual amt_out.
            """
            amt_in = int(amt_in)
            with pool.use_snapshot_context():
                amt_out_ = pool.trade(i, j, amt_in)
            return amt_out - amt_out_

        high = pool.get_max_trade_size(i, j)

        res = root_scalar(
            loss,
            args=(pool, i, j),
            bracket=(0, high),
            xtol=1e-6,
            method="brentq",
        )

        if res.converged:
            return int(res.root)
        else:
            raise RuntimeError(res.message)
