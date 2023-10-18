from .controller import Controller
from .utils import external_swap

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

    def arbitrage(self, llamma, spot_price):
        """
        @notice This is the soft liquidation
        TODO Need to account for fees
        """
        # if llamma.p == spot_price:
        #     return 0
        # elif llamma.p > spot_price:
        pass
