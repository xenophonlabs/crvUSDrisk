from collections import defaultdict
from .oracle import Oracle
from .utils import _plot_reserves

EPSILON = 1e-18 # to avoid division by 0
DEAD_SHARES = 1e-15 # to init shares in a band

class Swap:

    __slots__ = (
        'in_amount', 
        'out_amount',
        'reserves', # reserves[n] = [x, y]
        'admin_fee', # admin_fee[n] = amt
        'n1', # band where swap starts
        'n2', # band where swap ends
    )

    def __init__(self):
        self.in_amount = 0
        self.out_amount = 0
        self.reserves = defaultdict(lambda: [0,0])
        self.admin_fee = 0
        self.n1 = 0
        self.n2 = 0
    
    def __repr__(self):
        return f"Swap(\nin_amount={self.in_amount}\n out_amount={self.out_amount}\nreserves={self.reserves}\nadmin_fee={self.admin_fee}\nn1={self.n1}\nn2={self.n2})"
    
    def __str__(self):
        return f"Swap(\nin_amount={self.in_amount}\n out_amount={self.out_amount}\nreserves={self.reserves}\nadmin_fee={self.admin_fee}\nn1={self.n1}\nn2={self.n2})"

class LLAMMA:

    __slots__ = (
        # === Parameters === #
        'base_price', # Price at contract creation
        'A', # Amplification factor (width) for bands
        'fee', # Fee charged on swaps
        'admin_fee', # Pct of fees that go to admin
        'MAX_TICKS', # Define as an input param (instead of hard-coded at 50), to potentially test

        # === State variables === #
        'bands_x',  # bands_x[n] = stablecoins in band n
        'bands_y',  # bands_y[n] = collateral in band n
        'user_shares',  # user_shares[user, n] = user's share of band n
        'total_shares',  # total_shares[n] = total shares in band n
        'active_band',  # band for current price
        'min_band',  # bands below this are empty
        'max_band',  # bands above this are empty
        'admin_fees_x', # admin fees collected in stablecoins
        'admin_fees_y', # admin fees collected in collateral

        # === Dependencies/Inputs === #
        'oracle', # Oracle object
    )

    def __init__(
            self, 
            A: int, 
            base_price: float, 
            oracle: Oracle,
            fee: float,
            admin_fee: float = 1,
            MAX_TICKS: int = 50,
        ) -> None:
        # Set parameters
        self.A = A
        self.base_price = base_price # TODO: eventually updated by interest rate
        self.fee = fee
        self.admin_fee = admin_fee
        self.MAX_TICKS = MAX_TICKS

        # Set state variables
        self.bands_x = defaultdict(float)
        self.bands_y = defaultdict(float)
        self.admin_fees_x = 0
        self.admin_fees_y = 0
        self.user_shares = defaultdict(lambda: defaultdict(float))
        self.total_shares = defaultdict(float)
        self.active_band = 0
        self.min_band = 0
        self.max_band = 0

        # Set dependencies
        self.oracle = oracle

    @property
    def p_o(self) -> float:
        # TODO: limit oracle changes and dynamic fee
        return self.oracle.price()

    # === Main Functions === #
    
    def _swap(
            self, 
            amt_in: float, 
            y_in: bool
        ) -> Swap:
        """
        @notice Sell at most amt_in to the AMM. This is a VIEW function.
        @param amt_in amount of tokens to swap in
        @param y_in whether collateral is being sold to the AMM (True) or bought from the AMM (False)
        @return Swap object
        TODO Account for skipping ticks
        """
        s = Swap()
        s.n1 = s.n2 = self.active_band

        in_amount_left = amt_in

        x, y = self.bands_x[s.n2], self.bands_y[s.n2]

        for _ in range(self.MAX_TICKS):

            n = s.n2
            s.reserves[n] = [x, y] # Init

            if y_in:
                # Going right ->
                
                if x != 0:
                    assert self.f(n) > 0
                    y_dest = (self.inv(n) / self.f(n) - self.g(n)) - y # Amt required to clear this band
                    dy = y_dest / (1 - self.fee) # Amt required to clear this band + fee

                    if dy >= in_amount_left:
                        # This is the last band
                        y_dest = in_amount_left * (1 - self.fee)
                        x_left = self.inv(n) / (self.g(n) + (y + y_dest)) - self.f(n) + EPSILON
                        assert 0 < x_left < x 
                        admin_fee = (in_amount_left - y_dest) * self.admin_fee
                        # Updates
                        s.in_amount = amt_in # Used all amt_in
                        s.out_amount += x - x_left
                        s.reserves[n] = [x_left, y + in_amount_left - admin_fee]
                        s.admin_fee += admin_fee
                        break

                    else:
                        # Go into next band
                        admin_fee = (dy - y_dest) * self.admin_fee # total admin fee paid
                        in_amount_left -= dy
                        # Updates
                        s.out_amount += x
                        s.in_amount += dy
                        s.admin_fee += admin_fee
                        s.reserves[n] = [0, y + dy - admin_fee]

                if s.n2 == self.min_band:
                    # there is no more liquidity
                    break

                # Prepare next loop
                s.n2 -= 1
                x = self.bands_x[s.n2]
                y = 0 

            else:
                # Going left <-

                if y != 0:
                    assert self.g(n) > 0
                    x_dest = (self.inv(n) / self.g(n) - self.f(n)) - x # Amt required to clear this band
                    dx = x_dest / (1 - self.fee) # Amt required to clear this band + fee

                    if dx >= in_amount_left:
                        # This is the last band
                        x_dest = in_amount_left * (1 - self.fee)
                        y_left = self.inv(n) / (self.f(n) + (x + x_dest)) - self.g(n) + EPSILON
                        assert 0 < y_left < y
                        admin_fee = (in_amount_left - x_dest) * self.admin_fee
                        # Updates
                        s.in_amount = amt_in # Used all amt_in
                        s.out_amount += y - y_left
                        s.reserves[n] = [x + in_amount_left - admin_fee, y_left]
                        s.admin_fee += admin_fee
                        break

                    else:
                        # Go into next band
                        admin_fee = (dx - x_dest) * self.admin_fee # total admin fee paid
                        in_amount_left -= dx
                        # Updates
                        s.out_amount += y
                        s.in_amount += dx
                        s.admin_fee += admin_fee
                        s.reserves[n] = [x + dx - admin_fee, 0]

                if s.n2 == self.max_band:
                    # there is no more liquidity
                    break

                # Prepare next loop
                s.n2 += 1
                x = 0
                y = self.bands_y[s.n2] 

        return s

    def swap(
            self, 
            amt_in: float, 
            y_in: bool
        ) -> tuple:
        """
        @notice Swap tokens in pool. This is a soft liquidation.
        @param amt amount of tokens to swap in
        @param y_in whether collateral is being sold to the AMM (True) or bought from the AMM (False)
        @return [amt_in, amt_out] actual amount swapped in and out
        TODO add slippage tolerance
        """
        assert amt_in > 0

        # NOTE: amt is amount to swap IN
        s = self._swap(amt_in, y_in)
        
        if s.in_amount == 0 or s.out_amount == 0:
            return 0

        if y_in:
            self.admin_fees_y += s.admin_fee
        else:
            self.admin_fees_x += s.admin_fee

        for n, r in s.reserves.items():
            self.bands_x[n] = r[0]
            self.bands_y[n] = r[1]

        self.active_band = s.n2

        return (s.in_amount, s.out_amount)

    def deposit(
            self, 
            user: str, 
            amount: float, 
            n1: int, 
            n2: int
        ) -> None:
        N = n2 - n1 + 1
        assert N <= self.MAX_TICKS, "Too many ticks"
        assert self.user_shares[user] == defaultdict(float), "User already has shares"
        yn = amount / N
        for ni in range(n1, n2+1):
            assert self.bands_x[n1] == 0
            ds = (self.total_shares[ni] + DEAD_SHARES)*yn/(self.bands_y[ni] + EPSILON)
            assert ds > 0
            self.user_shares[user][ni] += ds
            self.total_shares[ni] += ds
            self.bands_y[ni] += yn
        
        self.min_band = min(self.min_band, n1)
        self.max_band = max(self.max_band, n2)

    def withdraw(
            self, 
            user: str, 
            frac: float
        ) -> tuple:
        """
        @notice Evenly withdraw frac of the user's shares from each band.
        @param user user address
        @param frac fraction of shares to withdraw
        @return [x, y] [amount of stablecoins withdrawn, amount of collateral withdrawn]
        """
        assert frac <= 1
        user_bands = self.user_shares[user].keys()
        assert len(self.user_shares[user]) >= 0
        n1 = min(user_bands)
        n2 = max(user_bands)

        total_x = 0
        total_y = 0
        min_band = self.min_band
        max_band = n1 - 1

        for n in range(n1, n2+1):

            x = self.bands_x[n]
            y = self.bands_y[n]

            ds = frac * self.user_shares[user][n]  
            self.user_shares[user][n] -= ds
            s = self.total_shares[n]
            new_shares = s - ds
            self.total_shares[n] = new_shares
            s += DEAD_SHARES
            dx = (x + EPSILON) * ds / s
            dy = (y + EPSILON) * ds / s

            x -= dx
            y -= dy
            
            if new_shares == 0:
                assert x == 0 & y == 0 

            if n == min_band:
                if x == 0 and y==0:
                    min_band += 1
            if x > 0 or y > 0:
                max_band = n

            self.bands_x[n] = x
            self.bands_y[n] = y
            total_x += dx
            total_y += dy

        # Empty the ticks
        if frac == 1:
            del self.user_shares[user]

        self.min_band = min_band
        if self.max_band <= n2:
            self.max_band = max_band

        # TODO: update rate
        # self.rate_mul = self._rate_mul()
        # self.rate_time = block.timestamp

        return (total_x, total_y)

    def get_x_down(self, user: str) -> float:
        """
        @notice calculate the amount of stablecoins obtainable
        from a user's position if prices decrease adiabatically.
        @param user user address
        @return x_down amount of stablecoins obtainable
        """
        user_bands = self.user_shares[user].keys()
        n1, n2 = min(user_bands), max(user_bands)
        x_down = 0

        for n in range(n1, n2+1):

            share = self.user_shares[user][n] / self.total_shares[n]

            I = self.inv(n)
            f = self.f(n)
            g = self.g(n)

            if self.p_o > self.p_o_up(n):

                y_o = I/f - g
                assert y_o > 0
                x_down += y_o * self.p_o_up(n) * ((self.A - 1)/self.A) ** 0.5 * share

                print(self.bands_y[n], y_o * self.p_o_up(n) * ((self.A - 1)/self.A) ** 0.5 * share)
            
            elif self.p_o < self.p_o_down(n):

                x_o = I/g - f
                assert x_o > 0
                x_down += x_o * share
            
            else:

                y_o = self.A * self.y0(n) * (self.p_o - self.p_o_down(n)) / self.p_o
                x_o = I/(g + y_o) - f
                assert y_o > 0
                assert x_o > 0
                x_down += x_o + y_o * (self.p_o_down(n) * self.p_o) ** 0.5 * share
        
        return x_down
            
    # === Helper Functions === #

    def _p(
            self, 
            n: int, 
            x: float, 
            y: float
        ) -> float:
        if x==0 and y==0:
            # return mid-price between p_c_up and p_c_down
            return self.p_c_down(n) * (self.A/(self.A-1))
        elif x==0:
            # lowest possible price of band
            return self.p_c_down(n)
        elif y==0:
            # highest possible price of band
            return self.p_c_up(n)
        return (x + self.f(n)) / (y + self.g(n))
    
    def p(self) -> float:
        """
        @notice wrapper to get price at current band
        """
        n = self.active_band
        return self._p(n, self.bands_x[n], self.bands_y[n])
    
    def p_o_up(self, n: int) -> float:
        return self.base_price * ((self.A-1)/self.A)**n
    
    def p_o_down(self, n: int) -> float:
        return self.p_o_up(n+1)
    
    def p_c_up(self, n: int) -> float:
        return self.p_c_down(n+1)
    
    def p_c_down(self, n: int) -> float:
        return self.p_o ** 3 / self.p_o_up(n) ** 2

    def f(self, n: int) -> float:
        return self.A * self.y0(n) * self.p_o**2 / self.p_o_up(n)

    def g(self, n: int) -> float:
        return (self.A - 1) * self.y0(n) * self.p_o_up(n) / self.p_o
    
    def inv(self, n: int) -> float:
        return (self.bands_x[n] + self.f(n)) * (self.bands_y[n] + self.g(n))

    def inv_up(self, n: int) -> float:
        return self.p_o * self.A**2 * self.y0(n)**2
    
    def band_width(self, n: int) -> float:
        return self.p_o_up(n) / self.A
    
    def _y0(
            self,
            x: float, 
            y: float, 
            p_o: float, 
            p_o_up: float
        ) -> float:
        # solve quadratic:
        # p_o * A * y0**2 - y0 * (p_oracle_up/p_o * (A-1) * x + p_o**2/p_oracle_up * A * y) - xy = 0
        a = p_o * self.A
        b = p_o_up * (self.A-1) * x / p_o + p_o**2/p_o_up * self.A * y
        c = x * y
        return (b + (b**2 - 4*a*c)**0.5) / (2*a)
    
    def y0(self, n: int):
        """
        @notice wrapper to get _y0 for input band
        """
        return self._y0(self.bands_x[n], self.bands_y[n], self.p_o, self.p_o_up(n))
    
    # === Plotting Functions === #

    def plot_reserves(self):
        """
        @notice Plot reserves in each band
        NOTE: for now, assume collateral price is = oracle price, and crvUSD price = $1
        """
        _plot_reserves(self)