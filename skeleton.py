class LLAMMA:

    def __init__(self, A, base_price, fee, oracle):
        self.bands = dict() # self.bands[n] = [x_n, y_n]
        self.shares = dict() # self.shares[user, n] = user's share of band n
        self.A = A
        self.base_price = base_price # TODO: eventually updated by interest rate
        self.fee = fee
        self.oracle = oracle
        self.p_o = oracle.price()

    def swap(self, amt_in, amt_out, tkn_in, tkn_out):
        """
        @notice This is a soft liquidation.
        """
        pass

    def deposit(self):
        pass

    def withdraw(self):
        pass

class Controller:

    def __init__(self):
        pass

    def health(self, user):
        pass

    def liquidate(self, user):
        pass

    def deposit(self, user, amount, N):
        pass

    def withdraw(self, user, frac):
        pass

    def repay(self, user, amount):
        pass

class Oracle:

    def __init__(self):
        pass

    def update(self):
        pass

    def price(self):
        self.update()
        return self.price

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

def sim(
        T, # number of timesteps
    ):
    # NOTE: For now assume Gas is 0? But DO create the Gas variable, and set it to 0.
    # NOTE: Eventually optimize as just DataFrame operations.

    dfs = [] # Store data from run

    # Generate collateral distribution <- This means Borrowers will create loans
    llamma = LLAMMA()
    controller = Controller()
    controller.create() # NOTE: Might want to just query subgraph?

    prices = [] # Gen from GBM. The price is collateral/USD price
    # TODO: Eventually, this price will be a function of Collateral/USDC and Collateral/USDT -> crvUSD/USD (from PKs) -> turn into Collateral/USD
    # ETH/USDC and crvUSD/USDC (Tricrypto) -> ETH/crvUSD (PK pool)
    # ETH/USDT and crvUSD/USDT (Tricrypto) -> ETH/crvUSD (PK pool)
    # LWA -> p = ETH/crvUSD
    # p_s = Aggregator price crvUSD/USD (all PK pools)
    # p = p * p_s -> ETH/USD <- this is the oracle price
    # for now, just generate ETH/USD from GBM?
    # Ultimately, will need to generate 6 price paths (+ crvUSD price path?)

    liquidity = [] # Create external slippage Curve for ETH and crvUSD?

    for t in range(T):
        # This loops through timesteps

        # Get price

        # First: update Peg Keepers. For now: pass <- this involves arbitrage and the update() function
        # This mints/burns crvUSD

        # Update oracle price <- This updates position healths

        # Liquidators liquidate positions or arbitrage LLAMMA <- This updates LLAMMA/Controller
        # NOTE: Liquidators do whatever is most profitable < check hard liquidations first, then arbs (soft liquidations)
        # NOTE: This is where slippage/liquidity is important

        # Borrowers update positions or create new loans <- This updates LLAMMA/Controller
        # TODO: How will borrowers update positions?
        # Try to have distribution be fixed (e.g. Normally around current price)

        # Update metrics in dfs <- e.g., calculate loss/bad debt
        pass
