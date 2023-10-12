import llamma as lm
import controller as cntrlr
import oracle as orcl
import liquidator as lqdtr

def sim(
        T, # number of timesteps
    ):
    # NOTE: For now assume Gas is 0? But DO create the Gas variable, and set it to 0.
    # NOTE: Eventually optimize as just DataFrame operations.

    dfs = [] # Store data from run

    # Generate collateral distribution <- This means Borrowers will create loans
    llamma = lm.LLAMMA(A=100,base_price=1800,oracle=orcl.Oracle(),fee=0.01,admin_fee=0.01)
    controller = cntrlr.Controller()
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
        print(t)


def main():
    sim(365)

if __name__ == "__main__":
    main()