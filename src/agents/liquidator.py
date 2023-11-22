from typing import List
from scipy.optimize import root_scalar
import logging
from .agent import Agent
from ..modules.market import ExternalMarket
from ..types.cycle import Swap, Liquidation, Cycle
from crvusdsim.pool.crvusd.controller import Position
from crvusdsim.pool.sim_interface import SimController
from crvusdsim.pool.sim_interface.sim_stableswap import SimCurveStableSwapPool
from ..utils import get_crvUSD_index
from dataclasses import dataclass


@dataclass
class Path:
    basis_token: str # TODO use Token class or something
    crvusd_pool: SimCurveStableSwapPool
    collat_pool: ExternalMarket


class Liquidator(Agent):
    """
    Liquidator performs hard liquidations on LLAMMAs.
    """

    liquidation_profit: float = 0
    liquidation_count: float = 0
    arbitrage_profit: float = 0
    arbitrage_count: float = 0

    basis_tokens: list = [
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC
        "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
    ]

    paths: List[Path] = []

    def __init__(self, tolerance: float = 0):
        assert tolerance >= 0
        self.tolerance = tolerance

    def set_paths(
        self,
        controller: SimController,
        crvUSD_pools: List[SimCurveStableSwapPool],
        collat_pools: List[ExternalMarket],
    ):
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

        if len(to_liquidate) == 0:
            return 0, 0

        underwater_debt = 0
        total_profit = 0

        for position in to_liquidate:
            profit = self.maybe_liquidate(position, controller)

            if profit > self.tolerance:
                total_profit += profit
                self.liquidation_count += 1
            else:
                # Liquidation was missed
                underwater_debt += position.debt

        self.liquidation_profit += total_profit

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

        to_repay = controller.tokens_to_liquidate(user)
        # TODO if to_repay == 0: perform liquidation.
        _, y = controller.AMM.get_sum_xy(user)

        result = {}
        for path in self.paths:
            crvusd_pool = path.crvusd_pool
            collat_pool = path.collat_pool

            # basis token -> crvUSD
            j = get_crvUSD_index(crvusd_pool)
            i = j ^ 1
            amt_in = self.search(crvusd_pool, i, j, to_repay)
            trade1 = Swap(crvusd_pool, i, j, amt_in)

            # crvUSD -> collateral
            trade2 = Liquidation(controller, user, to_repay)

            # collateral -> basis token
            amt_out = collat_pool.trade(0, 1, y)  # Expected amount out
            trade3 = Swap(collat_pool, 0, 1, y)

            expected_profit = amt_out - amt_in
            cycle = Cycle([trade1, trade2, trade3], expected_profit=expected_profit)
            result[path] = {"expected_profit": expected_profit, "cycle": cycle}

        # TODO should profit be marked to dollars? currently in basis token
        best = result[max(result, key=lambda k: result[k]["expected_profit"])]

        if best.expected_profit > self.tolerance:
            logging.info(
                f"Liquidating user {user} with expected profit: {best.expected_profit}."
            )
            profit = best.cycle.execute()
            return profit
        else:
            logging.info(f"Missed liquidation for user {user} with health {health}.")
            return 0

    def search(self, pool: SimCurveStableSwapPool, i: int, j: int, amt_out: int) -> int:
        """
        Find the amt_in required to get the desired
        amt_out from a swap.

        Currently only meant for USDC or USDT ->
        crvUSD.
        """

        assert isinstance(pool, SimCurveStableSwapPool)

        def loss(amt_in: float, pool: SimCurveStableSwapPool, i: int, j: int):
            """
            Loss function for optimization. Very simple:
            we just want to minimize the diff between the
            desired amt_out, and the actual amt_out.
            """
            with pool.use_snapshot_context():
                amt_out_ = pool.trade(i, j, amt_in)
            print(amt_in, amt_out, amt_out_)
            return abs(amt_out - amt_out_)

        high = pool.get_max_trade_size(i, j)
        
        res = root_scalar(
            loss,
            args=(pool, i, j),
            bracket=(0, high),
            xtol=1e-6,
            method="brentq",
        )

        if res.success:
            return res.root
        else:
            raise RuntimeError(res.message)
