from ..modules.controller import Controller, Position
from ..modules.llamma import LLAMMA
from ..utils import external_swap
from typing import List
import matplotlib.pyplot as plt
import copy
import numpy as np


class Liquidator:
    """
    Liquidator either hard or soft liquidates LLAMMA positions.
    TODO ensure pnl from hard and soft liquidations is in USD (NOT crvUSD) units!
    """

    __slots__ = (
        "tolerance",  # min profit required to liquidate (could be returns)
        "liquidation_pnl",
        "liquidation_count",
        "arbitrage_pnl",
        "arbitrage_count",
        "verbose",  # print statements
    )

    def __init__(
        self,
        tolerance: float,
        verbose: bool = False,
    ) -> None:
        assert tolerance >= 0

        self.tolerance = tolerance
        self.liquidation_pnl = 0
        self.liquidation_count = 0
        self.arbitrage_pnl = 0
        self.arbitrage_count = 0
        self.verbose = verbose

    # === Hard Liquidations === #

    def maybe_liquidate(
        self,
        controller: Controller,
        position: Position,
        ext_stable_liquidity: float,
        ext_collat_liquidity: float,
        ext_swap_fee: float,
    ) -> float:
        """
        This is the hard liquidation. Liquidator will compute
        the pnl from the liquidation, accounting for external
        slippage. We assume their quote asset is USDC which
        they obtain from the ETH/USDC pool on Uniswap.

        Parameters
        ----------
        controller : Controller
            Controller object
        position : Position
            Position object to liquidate
        ext_stable_liquidity : float
            External stablecoin liquidity (e.g. USDC in ETH/USDC pool)
        ext_collat_liquidity : float
            External collateral liquidity (e.g. ETH in ETH/USDC pool)
        ext_swap_fee : float
            External swap fee

        Returns
        -------
        float
            pnl in USDC units

        Note
        ----
        Ultimately we want to simulate pnl from the following path (using ETH as example):
        1. Flashswap ETH -> USDC
        2. Swap USDC -> crvUSD
        3. Liquidate (repay crvUSD debt) and receive ETH
        4. Repay flashswap (i.e. sell ETH)
        5. Swap remaining ETH for USDC.
        6. Swap remaining crvUSD for USDC.
        TODO incorporate crvUSD slippage
        TODO could convert the external_swap functionality into an ExternalMarket class or something
        """
        user = position.user
        health = position.health

        x_pnl, y_pnl = controller.check_liquidate(user, 1)

        # Sell y_pnl collateral for USD
        y_pnl = external_swap(
            ext_stable_liquidity, ext_collat_liquidity, y_pnl, ext_swap_fee, True
        )
        pnl = x_pnl + y_pnl
        if pnl > self.tolerance:
            # TODO need to account for gas
            controller.liquidate(user, 1)
            self.print(f"Liquidated user {user} with pnl {pnl}.")
            return pnl
        else:
            self.print(f"Missed liquidation for user {user} with health {health}.")
            return 0

    def perform_liquidations(
        self,
        controller: Controller,
        ext_stable_liquidity: float,
        ext_collat_liquidity: float,
        ext_swap_fee: float,
    ) -> List[float, float]:
        """
        Loops through all liquidatable users and liquidates if profitable.

        Parameters
        ----------
        controller : Controller
            Controller object.
        ext_stable_liquidity : float
            External stablecoin liquidity (e.g. USDC in ETH/USDC pool).
        ext_collat_liquidity : float
            External collateral liquidity (e.g. ETH in ETH/USDC pool).
        ext_swap_fee : float
            External swap fee.

        Returns
        -------
        total_pnl : float
            PnL in USDC units from Liquidations.
        underwater_debt : float
            Total crvUSD debt of underwater positions.
        """
        to_liquidate = controller.users_to_liquidate()

        if len(to_liquidate) == 0:
            return 0, 0

        underwater_debt = 0
        total_pnl = 0

        for position in to_liquidate:
            pnl = self.maybe_liquidate(
                controller,
                position,
                ext_stable_liquidity,
                ext_collat_liquidity,
                ext_swap_fee,
            )

            if pnl > 0:
                total_pnl += pnl
                self.liquidation_count += 1

            else:
                # Liquidation was missed
                underwater_debt += position.debt

        self.liquidation_pnl += total_pnl

        return total_pnl, underwater_debt

    # === Soft Liquidations === #

    def arbitrage(
        self,
        amm: LLAMMA,
        p_mkt: float,
        ext_stable_liq: float,
        ext_collat_liq: float,
        ext_swap_fee: float,
    ) -> float:
        """
        Calculate optimal arbitrage and perform it.

        Parameters
        ----------
        amm : LLAMMA
            LLAMMA object.
        p_mkt : float
            Market price.
        ext_stable_liquidity : float
            External stablecoin liquidity (e.g. USDC in ETH/USDC pool).
        ext_collat_liquidity : float
            External collateral liquidity (e.g. ETH in ETH/USDC pool).
        ext_swap_fee : float
            External swap fee.

        Returns
        -------
        float
            pnl in crvUSD units.
        """
        p_opt, pnl = Liquidator.get_optimal_arb(
            amm, p_mkt, ext_stable_liq, ext_collat_liq, ext_swap_fee
        )
        amt_in, _, pump = Liquidator.calc_arb_amounts(amm, p_opt)

        if amt_in > 0 and pnl > self.tolerance:
            self.print(f"Performed arbitrage, profit: {round(pnl)} USD")
            amm.swap(amt_in, not pump)
            self.arbitrage_pnl += pnl
            self.arbitrage_count += 1

        return max(pnl, 0)

    @staticmethod
    def calc_arb_amounts(amm: LLAMMA, p: float) -> List[float, float, bool]:
        """
        Calculates the swap_in and swap_out for an arbitrageur
        moving LLAMMA price to p.

        Parameters
        ----------
        amm : LLAMMA
            LLAMMA object.
        p : float
            New price to move AMM to.

        Returns
        -------
        amt_in : float
            Amount of token in to push price to p.
        amt_out : float
            Amount of token out when pushing price to p.
        pump : bool
            True if stablecoin is being sold to AMM, False otherwise.
        """

        if p == amm.p:
            # Do nothing
            return 0, 0, False

        pump = (
            True if p > amm.p else False
        )  # pump == True means stablecoin being sold to the AMM

        amt_x = 0
        amt_y = 0

        n = amm.active_band

        for _ in range(amm.MAX_TICKS):
            not_empty = amm.bands_x[n] > 0 or amm.bands_y[n] > 0

            if p <= amm.p_c_up(n) and p >= amm.p_c_down(n):
                if not_empty:
                    new_x = max((amm.inv(n) * p) ** 0.5, amm.f(n)) - amm.f(n)
                    new_y = max((amm.inv(n) / p) ** 0.5, amm.g(n)) - amm.g(n)
                    delta_x = new_x - amm.bands_x[n]
                    delta_y = new_y - amm.bands_y[n]
                    if pump:
                        delta_y *= -1
                    else:
                        delta_x *= -1
                    amt_x += delta_x
                    amt_y += delta_y
                break
            else:
                # We clear this band, so the user either gets all collateral or all the stablecoin in the band
                if pump:
                    if not_empty:
                        amt_x += amm.inv(n) / amm.g(n) - amm.f(n) - amm.bands_x[n]
                        amt_y += amm.bands_y[n]
                    n += 1
                else:
                    if not_empty:
                        amt_x += amm.bands_x[n]
                        amt_y += amm.inv(n) / amm.f(n) - amm.g(n) - amm.bands_y[n]
                    n -= 1

        if pump:
            # we are going left <- selling stablecoin to AMM
            amt_in = amt_x / (1 - amm.fee)
            amt_out = amt_y
        else:
            # we are going right -> selling collateral to AMM
            amt_in = amt_y / (1 - amm.fee)
            amt_out = amt_x

        return amt_in, amt_out, pump

    @staticmethod
    def arb_profits(
        amm: LLAMMA,
        p_new: float,
        ext_stable_liquidity,
        ext_collat_liquidity,
        ext_swap_fee,
    ) -> float:
        """
        Calculate arbitrage profits from moving AMM price to p_new, accounting
        for external slippage.

        Parameters
        ----------
        LLAMMA object
        p_new : float
            Price to move AMM to
        ext_stable_liquidity : float
            External stablecoin liquidity (e.g. USDC in ETH/USDC pool)
        ext_collat_liquidity : float
            External collateral liquidity (e.g. ETH in ETH/USDC pool)
        ext_swap_fee : float
            External swap fee

        Returns
        -------
        float
            pnl in crvUSD units
        """
        amt_in, amt_out, pump = Liquidator.calc_arb_amounts(amm, p_new)
        if amt_in == 0 or amt_out == 0:
            return 0

        Liquidator.test_arb(amm, p_new)  # not necessary

        amt_out = external_swap(
            ext_stable_liquidity, ext_collat_liquidity, amt_out, ext_swap_fee, pump
        )
        pnl = amt_out - amt_in

        # TODO Ensure arb profits in USD units
        if pump:
            # we are selling stablecoin to AMM, pnl in units of stablecoin
            pnl *= 1
        else:
            ext_p = ext_stable_liquidity / ext_collat_liquidity
            pnl *= ext_p  # convert collateral units to crvUSD units

        return pnl

    @staticmethod
    def get_optimal_arb(
        amm: LLAMMA,
        p_mkt: float,
        ext_stable_liquidity: float,
        ext_collat_liquidity: float,
        ext_swap_fee: float,
        plot: bool = False,
    ) -> tuple[float, float]:
        """
        Search for optimal arbitrage price and pnl.

        Parameters
        ----------
        amm : LLAMMA
            LLAMMA object
        p_mkt : float
            Market price
        ext_stable_liquidity : float
            External stablecoin liquidity (e.g. USDC in ETH/USDC pool)
        ext_collat_liquidity : float
            External collateral liquidity (e.g. ETH in ETH/USDC pool)
        ext_swap_fee : float
            External swap fee
        plot : bool, optional
            Whether to plot the profit curve, by default False.

        Returns
        -------
        float
            Optimal price to move AMM to
        float
            PnL for performing optimal arbitrage

        Note
        ----
        TODO If this needs to be faster we can:
        1. Implement a better search algo
        2. Derive the optimal p_new analytically <- this is hard
        """
        # Expanding the interval with 0.9 and 1.1 for plotting
        p_low = min(amm.p, p_mkt) * 0.9
        p_high = max(amm.p, p_mkt) * 1.1

        ps = np.linspace(p_low, p_high, 100)

        profits = [
            Liquidator.arb_profits(
                amm, p, ext_stable_liquidity, ext_collat_liquidity, ext_swap_fee
            )
            for p in ps
        ]

        max_profits = max(profits)
        idx = profits.index(max_profits)

        if plot:
            plt.plot(ps, profits)
            plt.axvline(ps[idx], color="black", linestyle="--")

        if max_profits == 0:
            return amm.p, 0  # Don't do anything

        return ps[idx], max_profits

    def test_arb(amm: LLAMMA, p_new):
        """
        Ensures the stateless `calc_arb_amounts` function matches
        the stateful `swap` function in LLAMMA.

        Parameters
        ----------
        amm : LLAMMA
            LLAMMA object
        p_new : float
            New price to move AMM to

        Raises
        ------
        AssertionError
            If the swap amounts do not match
        """
        amm_cp = copy.deepcopy(amm)
        amt_in, amt_out, pump = Liquidator.calc_arb_amounts(amm_cp, p_new)
        if amt_in != 0 and amt_out != 0:
            _, swap_out = amm_cp.swap(amt_in, not pump)
            assert abs(swap_out - amt_out) < 1e-6, AssertionError(
                "Arb optimization failed to match LLAMMA."
            )  # FIXME sometimes you just can't reach the target price

    def print(self, txt):
        if self.verbose:
            print(txt)
