class Liquidator:

    def __init__(self):
        pass

    def liquidate(self, controller, user):
        """
        @notice This is the hard liquidation
        """
        controller.liquidate(user)

    def arbitrage(self):
        """
        @notice This is the soft liquidation
        """
        pass
