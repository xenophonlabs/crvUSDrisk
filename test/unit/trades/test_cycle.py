"""
Test the `Cycle` class, which is responsible for
all the trading logic in the model.
"""

"""
df = pd.DataFrame(
    columns=["amt_optimize", "amt_linspace", "profit_optimize", "profit_linspace"]
)

for i, cycle in enumerate(cycles):
    try:
        trade = cycle.trades[0]
        decimals = trade.pool.coin_decimals[trade.i]
        address = trade.pool.coin_addresses[trade.i].lower()

        # Set xatol
        xatol = int(10**decimals / sample.prices_usd[address])

        high = trade.pool.get_max_trade_size(trade.i, trade.j)
        amts = np.linspace(0, high, 1000)
        amts = [int(amt) for amt in amts]
        profits = [cycle.populate(amt) for amt in amts]

        best_amt_linspace = int(amts[np.argmax(profits)])
        best_profit_linspace = int(max(profits))

        best_amt_optimize, best_profit_optimize = cycle.optimize(xatol)

        df.loc[i] = [
            best_amt_optimize,
            best_amt_linspace,
            best_profit_optimize,
            best_profit_linspace,
        ]

    except Exception as e:
        print(f"Cycle {i} failed to optimize: {e}")

df["profit_linspace"] = df["profit_linspace"].astype(float)
df["profit_optimize"] = df["profit_optimize"].astype(float)
df = df.round(3)
df
"""
