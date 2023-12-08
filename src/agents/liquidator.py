"""Provides the `Liquidator` class."""
import math
from typing import List, Tuple
from dataclasses import dataclass
from scipy.optimize import root_scalar
from crvusdsim.pool.crvusd.controller import Position
from crvusdsim.pool.sim_interface import SimController
from crvusdsim.pool.sim_interface.sim_stableswap import SimCurveStableSwapPool
from .agent import Agent
from ..modules import ExternalMarket
from ..trades.cycle import Swap, Liquidation, Cycle
from ..utils import get_crvusd_index
from ..configs import TOKEN_DTOs, DEFAULT_PROFIT_TOLERANCE
from ..types import MarketsType
from ..data_transfer_objects import TokenDTO
from ..logging import get_logger


logger = get_logger(__name__)


@dataclass
class Path:
    """Simple dataclass to store liquidation paths."""

    basis_token: TokenDTO
    crvusd_pool: SimCurveStableSwapPool
    collat_pool: ExternalMarket


class Liquidator(Agent):
    """
    The Liquidator performs hard liquidations on LLAMMAs.
    Liquidations are routed through `Path`s, which are
    set by the `set_paths` function. A Path will always
    be a `Cycle` of trades that begins and ends with a
    `basis_token`.

    Example
    -------
    With USDC as the basis token,
    a liquidation will look like:
    1. Purchase crvUSD from the crvUSD/USDC pool.
    2. Liquidate the user in the Controller.
    3. Sell collateral for USDC in the USDC/collateral External Market.

    TODO consider more general liquidation paths.
    """

    basis_tokens: List[TokenDTO] = [
        TOKEN_DTOs["0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"],  # USDC
        TOKEN_DTOs["0xdac17f958d2ee523a2206206994597c13d831ec7"],  # USDT
    ]

    paths: List[Path] = []

    def __init__(self, tolerance: float = DEFAULT_PROFIT_TOLERANCE):
        assert tolerance >= 0
        self.tolerance = tolerance
        self._profit: float = 0.0
        self._count: int = 0

    def set_paths(
        self,
        controller: SimController,
        crvusd_pools: List[SimCurveStableSwapPool],
        collat_pools: MarketsType,
    ):
        """
        Set the paths for liquidations. Currently:
        1. Purchase crvusd from basis_token/crvusd pools.
        2. Liquidate the user in the Controller.
        3. Sell collateral for basis_token in collateral/basis_token
        External markets.
        """
        collateral = TOKEN_DTOs[controller.COLLATERAL_TOKEN.address]

        self.paths = []
        for basis_token in self.basis_tokens:
            pair = (
                min(basis_token, collateral),
                max(basis_token, collateral),
            )  # sorted

            # Get basis_token/crvusd pool
            for pool in crvusd_pools:
                coins = [c.address for c in pool.coins]
                if basis_token.address in coins:
                    crvusd_pool = pool
                    # TODO assuming only one pool for each basis token
                    break

            # Get collateral/basis_token pool
            collat_pool = collat_pools[pair]
            self.paths.append(Path(basis_token, crvusd_pool, collat_pool))

    def perform_liquidations(
        self,
        controller: SimController,
    ) -> Tuple[float, int]:
        """
        Loops through all liquidatable users and liquidates if profitable.

        Parameters
        ----------
        controller : Controller
            Controller object.

        Returns
        -------
        profit : float
            Profit in USD units from liquidations.
        count : int
            Count of liquidations performed.

        TODO liquidations should be ordered by profitability
        """
        to_liquidate = controller.users_to_liquidate()

        logger.info("There are %d users to liquidate.", len(to_liquidate))

        if len(to_liquidate) == 0:
            return 0.0, 0

        profit = 0.0
        count = 0

        for position in to_liquidate:
            profit_ = self.maybe_liquidate(position, controller)

            if profit_ > self.tolerance:
                profit += profit_
                count += 1

        self._profit += profit
        self._count += count

        return profit, count

    # pylint: disable=too-many-locals
    def maybe_liquidate(
        self,
        position: Position,
        controller: SimController,
    ) -> float:
        """
        This is the hard liquidation:
        1. Liquidator checks the crvusd debt they'll have to repay.
        2. For each basis token (e.g. USDC, USDT) they:
            a. Compute how much of the basis token they must swap
            to obtain the necessary crvusd.
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
        TODO incorporate liquidations that source crvusd partly from
        USDC and partly from USDT.
        TODO incorporate liquidations against other tokens (not just
        USDC and USDT).
        TODO use the ERC20 dataclass for all token objects in codebase
        """
        user = position.user
        health = position.health

        # TODO int casting should occur in controller
        to_repay = int(controller.tokens_to_liquidate(user))
        _, y = controller.AMM.get_sum_xy(user)
        y = int(y)

        best = None
        best_expected_profit = -math.inf

        collateral = controller.COLLATERAL_TOKEN.address

        assert self.paths, "Liquidator paths not set."
        for path in self.paths:
            crvusd_pool = path.crvusd_pool
            collat_pool = path.collat_pool

            # TODO Abstract this into the `Cycle.populate` function
            # TODO dollarize the profit based on current USD market prices

            # basis token -> crvusd
            j = get_crvusd_index(crvusd_pool)
            i = j ^ 1
            amt_in = crvusd_pool.get_dy(i, j, to_repay)
            trade1 = Swap(crvusd_pool, i, j, amt_in)

            # crvusd -> collateral
            trade2 = Liquidation(controller, position, to_repay)

            # collateral -> basis token
            i = collat_pool.coin_addresses.index(collateral)
            j = i ^ 1
            trade3 = Swap(collat_pool, i, j, y)
            amt_out, decimals = trade3.do(use_snapshot_context=True)

            expected_profit = (amt_out - amt_in) / 10**decimals

            cycle = Cycle([trade1, trade2, trade3], expected_profit=expected_profit)

            if not best or expected_profit > best_expected_profit:
                best = cycle
                best_expected_profit = expected_profit

        if best and best_expected_profit > self.tolerance:
            logger.info(
                "Liquidating user %s with expected profit: %f.",
                user,
                best_expected_profit,
            )
            profit = best.execute()
            logger.info("Liquidated user %s with profit: %f.", user, profit)
            return profit
        logger.info(
            "Missed liquidation for user %s. Health: %f. Expected profit: %f.",
            user,
            health,
            best_expected_profit,
        )
        return 0.0

    def search(self, pool: SimCurveStableSwapPool, i: int, j: int, amt_out: int) -> int:
        """
        DEPRECATED: use `SimCurveStableSwapPool.get_dy` instead.

        Find the amt_in required to get the desired
        amt_out from a swap.

        This is essentially a more optimized binary search
        using Brent's method.
        """
        amt_out_float = float(amt_out)  # For scipy

        assert isinstance(pool, SimCurveStableSwapPool)

        def loss(amt_in: int, pool: SimCurveStableSwapPool, i: int, j: int):
            """
            Loss function for optimization. Very simple:
            we just want to minimize the diff between the
            desired amt_out, and the actual amt_out.
            """
            amt_in = int(amt_in)
            with pool.use_snapshot_context():
                _, amt_out_, _ = pool.trade(i, j, amt_in)
            return amt_out_float - amt_out_

        high = pool.get_max_trade_size(i, j)

        res = root_scalar(
            loss,
            args=(pool, i, j),
            bracket=(0, high),
            xtol=1,  # Absolute error. This is 1e-18 in crvusd units TODO could make =$1 MTM
            method="brentq",
            # maxiter=1000,
        )

        if res.converged:
            return int(res.root)
        raise RuntimeError(res.flag)
