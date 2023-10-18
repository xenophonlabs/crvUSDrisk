from .controller import Controller
from .llamma import LLAMMA
from .utils import external_swap
import matplotlib.pyplot as plt
import copy
import numpy as np

class Liquidator:

    __slots__ = (
        'tolerance', # min profit required to liquidate (could be returns)
        'pnl', # pnl from liquidations
    )

    def __init__(
            self, 
            tolerance,
        ) -> None:
        self.tolerance = tolerance
        self.pnl = 0

    def maybe_liquidate(
            self, 
            controller: Controller, 
            user: str,
            ext_stable_liquidity: float,
            ext_collat_liquidity: float,
            ext_swap_fee: float,
        ) -> float:
        """
        @notice This is the hard liquidation
        TODO incorporate crvUSD slippage
        NOTE for now assume crvUSD is 1:1 with USD w/ no slippage
        TODO could convert the external_swap functionality into an ExternalMarket class or something
        """
        x_pnl, y_pnl = controller.check_liquidate(user, 1)

        # Sell y_pnl collateral for USD
        y_pnl = external_swap(ext_stable_liquidity, ext_collat_liquidity, y_pnl, ext_swap_fee, True)
        pnl = x_pnl + y_pnl
        if pnl > self.tolerance:
            # TODO need to account for gas
            controller.liquidate(user, 1)
            print(f"Liquidated user {user} with pnl {pnl}.")
            return pnl
        else:
            print(f"Missed liquidation for user {user} with pnl {pnl}.")
            return 0
    
    def perform_liquidations(
            self, 
            controller: Controller,
            ext_stable_liquidity: float,
            ext_collat_liquidity: float,
            ext_swap_fee: float,
        ) -> None:
        """
        @notice loops through all liquidatable users and 
        liquidate if profitable
        """
        to_liquidate = controller.users_to_liquidate()

        if len(to_liquidate) == 0:
            print("No users to liquidate.")
            return
        
        for position in to_liquidate:
            self.pnl += self.maybe_liquidate(controller, position.user, ext_stable_liquidity, ext_collat_liquidity, ext_swap_fee)

    # === Soft Liquidations === #

    @staticmethod
    def calc_arb_amounts(amm: LLAMMA, p):
        """
        @notice Calculates the swap_in and swap_out for an arbitrageur
        moving LLAMMA price to p.
        """

        pump = True if p > amm.p else False # pump == True means stablecoin being sold to the AMM

        amt_x = 0 
        amt_y = 0

        n = amm.active_band

        for _ in range(amm.MAX_TICKS):

            not_empty = amm.bands_x[n] > 0 or amm.bands_y[n] > 0

            if p <= amm.p_c_up(n) and p >= amm.p_c_down(n):
                if not_empty:
                    new_x = max((amm.inv(n)*p)**0.5, amm.f(n)) - amm.f(n)
                    new_y = max((amm.inv(n)/p)**0.5, amm.g(n)) - amm.g(n)
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
                        amt_x += amm.inv(n)/amm.g(n) - amm.f(n) - amm.bands_x[n]
                        amt_y += amm.bands_y[n]
                    n += 1
                else:
                    if not_empty:
                        amt_x += amm.bands_x[n]
                        amt_y += amm.inv(n)/amm.f(n) - amm.g(n) - amm.bands_y[n]
                    n -= 1

        if pump:
            # we are going right -> selling collateral to AMM
            amt_in = amt_x / (1 - amm.fee)
            amt_out = amt_y
        else:
            # we are going left <- selling crvUSD to AMM
            amt_in = amt_y / (1 - amm.fee)
            amt_out = amt_x
        
        return amt_in, amt_out, pump
        
    @staticmethod
    def arb_profits(amm: LLAMMA, p_new: float, ext_stable_liquidity, ext_collat_liquidity, ext_swap_fee):
        """
        @notice returns profit in units of token in.
        """
        amt_in, amt_out, pump = Liquidator.calc_arb_amounts(amm, p_new)
        if amt_in == 0 or amt_out == 0:
            return 0
        
        Liquidator.test_arb(amm, p_new) # not necessary

        amt_out = external_swap(ext_stable_liquidity, ext_collat_liquidity, amt_out, ext_swap_fee, pump)

        return amt_out - amt_in
    
    @staticmethod
    def get_optimal_arb(llamma, p_mkt, ext_stable_liquidity, ext_collat_liquidity, ext_swap_fee):
        """
        Dumb linear search
        TODO Really need this to be faster.
        """
        p_low = min(llamma.p, p_mkt)*0.9
        p_high = max(llamma.p, p_mkt)*1.1

        ps = np.linspace(p_low, p_high, 1000)

        profits = [Liquidator.arb_profits(llamma, p, ext_stable_liquidity, ext_collat_liquidity, ext_swap_fee) for p in ps]

        idx = profits.index(max(profits))

        plt.plot(ps, profits)
        plt.axvline(ps[idx], color='black', linestyle='--')

        return ps[idx]
    
        """
        # Try binary search
        epsilon = 1e-3
        p_low = min(llamma.p, p_mkt)*0.5
        p_high = max(llamma.p, p_mkt)*1.5
        max_profit = 0
        best_p_new = p_low
        while p_low <= p_high:
            p_mid = (p_low + p_high) / 2
            profit = Liquidator.arb_profits(llamma, p_mid, ext_stable_liquidity, ext_collat_liquidity, ext_swap_fee)
            print(f'try: {p_mid, profit}')
            
            if profit > max_profit:
                max_profit = profit
                best_p_new = p_mid
            
            if profit > 0:
                p_low = p_mid + epsilon
            else:
                p_high = p_mid - epsilon
        """

    def test_arb(amm: LLAMMA, p_new):
        """
        TODO failure cases:
        1. No liquidity (on either side) means arbitrageurs can't arb
        """

        amm_cp = copy.deepcopy(amm)

        amt_in, amt_out, pump = Liquidator.calc_arb_amounts(amm_cp, p_new)
        if amt_in == 0 or amt_out == 0:
            print("No arb opportunity")
            return
        # if pump == True then stablecoin is being sold to the AMM

        swap_in, swap_out = amm_cp.swap(amt_in, not pump)

        assert abs(swap_out - amt_out) < 1e-6
        assert abs(p_new - amm_cp.p) < 1e-6
