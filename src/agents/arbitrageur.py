import logging
from scipy.optimize import minimize_scalar
from ..utils import get_crvUSD_index
from ..types import Trade
from .agent import Agent

PRECISION = 1e18


class Arbitrageur(Agent):
    """
    Arbitrageur performs cyclic arbitrages between the following
    Curve pools:
        - StableSwap pools
        - TriCrypto-ng pools TODO
        - LLAMMAs.
    TODO need to investigate which pools to include in arbitrage
    search (e.g. TriCRV, other crvUSD pools, etc..). Otherwise, we
    are artificially constraining the available crvUSD liquidity.
    """

    def __init__(self, tolerance: float = 0):
        assert tolerance >= 0

        self.tolerance = tolerance
        self._profit = 0
        self._count = 0

    def arbitrage(self, pools, prices):
        """
        Identify the optimal arbitrage between pools
        and perform it if profitable.

        Parameters
        ----------
        pools : List[SimCurvePool]
            list of curve pool objects
        prices : List[float]
            list of external market prices for underlying tokens

        Returns
        -------
        profit
        count
        volume

        Note
        ----
        FIXME need to handle >2 pools
        FIXME need to return metrics that are pool specific
        """
        assert len(pools) == 2, NotImplementedError("Can't arb more than two pools.")

        # Call search on all combinations of pools
        trade = self.search(
            pools, prices
        )  # <- this already tells you the most profitable trade for pools A <-> B

        count = 0
        profit = 0

        if trade.profit > self.tolerance:
            profit = self.trade(trade)  # Perform the trade
            assert profit > self.tolerance, RuntimeError("Trade unprofitable.")
            logging.info(f"Arbitrage trade with profit {round(profit)}.")
            count += 1

        self._profit += profit
        self._count += count

        return profit, count

    @staticmethod
    def profit(amt_in, pool1, pool2, p):
        """
        Calculate profit from trade. Accounts for external slippage,
        and assumes external venues are at p_mkt. We assume the
        following arbitrage path, based on flash swaps:

        If USDC/USD price crashes, and USDC/crvUSD pool is mispriced:

            1. Sell USDC and buy crvUSD from USDC/crvUSD pool.
            2. Sell crvUSD to USDT/crvUSD pool.
            3. USDT_out - USDC_in * USDC/USDT price = profit.

        This may be done more efficiently via a flash swap
        (we assume Uni v3 pools for flash swaps are not mispriced):

            1. Flash borrow USDC from WETH/USDC pool.
            2. Swap USDC for crvUSD on USDC/crvUSD pool.
            3. Swap crvUSD for USDT on USDT/crvUSD pool.
            4. Swap USDT for WETH to repay flashloan.
            5. Remaining USDT is profit.

        The below pseudo-code explains this at a high-level:
        profit = pool2.trade(pool1.trade(USDC_in, crvusd_in=False), crvusd_in=True)
            - USDC_in * USDC/USDT price

        We may then account for external (flash swap) slippage on
        either side of the above trade.

        Parameters
        ----------
        pool1 : SimCurvePool
            Pool to buy crvUSD from.
        pool2 : SimCurvePool
            Pool to sell crvUSD to.
        amt_in : float
            Amount of token in. Assumed not to be in 1e18 units to
            satisfy scipy's minimize_scalar().
        p : float
            Current market price of token_in/token_out.
            E.g. USDC/USDT if pool1 is USDC/crvUSD and pool2 is USDT/crvUSD.

        Note
        ----
        We are assuming that arbitrageurs don't have crvUSD inventory.
        TODO handle >2 pools.
        """
        amt_in = int(amt_in * PRECISION)  # convert to 1e18 units

        # amt_in is non crvUSD (e.g. USDC)
        coin_out = get_crvUSD_index(pool1)
        coin_in = coin_out ^ 1
        with pool1.use_snapshot_context():
            amt_mid, _ = pool1.trade(coin_in, coin_out, int(amt_in))

        # amt_mid is crvUSD
        coin_in = get_crvUSD_index(pool2)
        coin_out = coin_in ^ 1
        with pool2.use_snapshot_context():
            amt_out, _ = pool2.trade(coin_in, coin_out, int(amt_mid))

        # amt_out is non crvUSD (e.g. USDT)
        # FIXME incorporate slippage and convert to USD
        # E.g., arbitrageurs will sell the USDT profits at prevailing
        # USDT/USD market price.
        return (amt_out - amt_in * p) / 1e18

    @staticmethod
    def neg_profit(amt_in, pool1, pool2, p):
        """
        Simple negative wrapper for maximizing profit.
        """
        return -Arbitrageur.profit(amt_in, pool1, pool2, p)

    @staticmethod
    def search(pools, prices):
        """
        Find the optimal liquidity-constrained cyclic arbitrages.

        TODO handle >2 pools, including LLAMMAs. Must output a
        queue of trades to perform.
        """
        assert (
            len(pools) == len(prices) == 2
        )  # FIXME need to handle >2 pools, but maybe not here.

        logging.info("Searching for arbitrage opportunities.")

        # Assume we always want to sell the "cheap" token
        # because we assume the pool is mispriced.
        # FIXME might want to consider both directions

        p_mkt = prices[0] / prices[1]  # stablecoin0/stablecoin1

        crvUSD_idx_0 = get_crvUSD_index(pools[0])
        p_pool_0 = pools[0].price(crvUSD_idx_0 ^ 1, crvUSD_idx_0)  # stablecoin0/crvUSD

        crvUSD_idx_1 = get_crvUSD_index(pools[1])
        p_pool_1 = pools[1].price(crvUSD_idx_1 ^ 1, crvUSD_idx_1)  # stablecoin1/crvUSD

        p_pool = p_pool_0 / p_pool_1  # stablecoin0/stablecoin1

        if p_pool > p_mkt:
            # stablecoin0/stablecoin1 in Curve pool is too expensive
            # sell stablecoin0 and buy crvUSD from pool0
            # sell crvUSD and buy stablecoin1 from pool1
            args = (pools[0], pools[1], p_mkt)
            high = (
                pools[0].get_max_trade_size(crvUSD_idx_0 ^ 1, crvUSD_idx_0) / PRECISION
            )
        elif 1 / p_pool - 1 / p_mkt > 0:
            # stablecoin0/stablecoin1 in Curve pool is too cheap
            # sell stablecoin1 and buy crvUSD from pool1
            # sell crvUSD and buy stablecoin0 from pool0
            args = (pools[1], pools[0], 1 / p_mkt)
            high = (
                pools[1].get_max_trade_size(crvUSD_idx_1 ^ 1, crvUSD_idx_1) / PRECISION
            )
        else:
            raise ValueError("Not sure what happens here.")

        res = minimize_scalar(
            Arbitrageur.neg_profit, args=args, bounds=(0, high), method="bounded"
        )
        if res.success:
            n = 1  # TODO should be # of trades to perform
            logging.info(f"Identified {n} arbitrage opportunities.")
            amt_in = int(res.x * PRECISION)
            profit = -res.fun
            trade = Trade(amt_in, profit, *args)
            return trade
        else:
            raise RuntimeError(
                res.message
            )  # Could make a minimization failed error class

    def trade(self, trade: Trade, use_snapshot_context=False):
        """
        Perform trade.

        TODO trade needs to handle >2 hops.
        """
        amt_in, pool1, pool2, p = trade.unpack()

        final_coin_in = pool1.metadata["coins"]["names"][get_crvUSD_index(pool1) ^ 1]
        final_coin_out = pool2.metadata["coins"]["names"][get_crvUSD_index(pool2) ^ 1]

        # Sell stablecoin1 and buy crvUSD from pool1
        coin_in = pool1.metadata["coins"]["names"][get_crvUSD_index(pool1) ^ 1]
        coin_out = pool1.metadata["coins"]["names"][get_crvUSD_index(pool1)]
        logging.info(
            f"Swapping {round(amt_in / PRECISION)} {coin_in} in pool {pool1.metadata['name']}."
        )
        if use_snapshot_context:
            with pool1.use_snapshot_context():
                amt_mid, _ = pool1.trade(coin_in, coin_out, int(amt_in))
        else:
            amt_mid, _ = pool1.trade(coin_in, coin_out, int(amt_in))
        logging.info(
            f"Received {round(amt_mid / PRECISION)} {coin_out} from pool {pool1.metadata['name']}."
        )

        # Sell crvUSD and buy stablecoin2 from pool2
        coin_in = pool2.metadata["coins"]["names"][get_crvUSD_index(pool2)]
        coin_out = pool2.metadata["coins"]["names"][get_crvUSD_index(pool2) ^ 1]
        logging.info(
            f"Swapping {round(amt_mid / PRECISION)} {coin_in} in pool {pool2.metadata['name']}."
        )
        if use_snapshot_context:
            with pool2.use_snapshot_context():
                amt_out, _ = pool2.trade(coin_in, coin_out, int(amt_mid))
        else:
            amt_out, _ = pool2.trade(coin_in, coin_out, int(amt_mid))
        logging.info(
            f"Received {round(amt_out / PRECISION)} {coin_out} from pool {pool2.metadata['name']}."
        )

        _profit = amt_out - amt_in * p
        logging.info(
            f"Profit: {round(_profit / PRECISION)} {coin_out} at market price: {round(p, 4)} {final_coin_in}/{final_coin_out}."
        )

        return _profit / PRECISION
