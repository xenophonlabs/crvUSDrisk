from collections import defaultdict
import matplotlib.pyplot as plt

EPSILON = 1e-18 # to avoid division by 0
DEAD_SHARES = 1e-15 # to init shares in a band

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 10})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

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
            A, 
            base_price, 
            oracle,
            fee,
            admin_fee = 1,
            MAX_TICKS = 50,
        ):
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
    
    def _swap(self, amt_in, y_in) -> Swap:
        """
        @notice Sell at most amt_in to the AMM.
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

    def swap(self, amt_in, y_in):
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

        return s.in_amount, s.out_amount

    def deposit(self, user, amount, n1, n2):
        N = n2 - n1 + 1
        assert N <= self.MAX_TICKS, "Too many ticks"
        assert self.user_shares[user] == defaultdict(float), "User already has shares"
        yn = amount / N
        for ni in range(n1, n2):
            assert self.bands_x[n1] == 0
            ds = (self.total_shares[ni] + DEAD_SHARES)*yn/(self.bands_y[ni] + EPSILON)
            assert ds > 0
            self.user_shares[user][ni] += ds
            self.total_shares[ni] += ds
            self.bands_y[ni] += yn
        
        self.min_band = min(self.min_band, n1)
        self.max_band = max(self.max_band, n2)

    def withdraw(self, user, frac):
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

        return [total_x, total_y]

    @property
    def p_o(self):
        # TODO: limit oracle changes and dynamic fee
        return self.oracle.price()

    # === Helper Functions === #

    def _p(self, n, x, y):
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
    
    def p(self):
        """
        @notice wrapper to get price at current band
        """
        n = self.active_band
        return self._p(n, self.bands_x[n], self.bands_y[n])
    
    def p_o_up(self, n):
        return self.base_price * ((self.A-1)/self.A)**n
    
    def p_o_down(self, n):
        return self.p_o_up(n+1)
    
    def p_c_up(self, n):
        return self.p_c_down(n+1)
    
    def p_c_down(self, n):
        return self.p_o ** 3 / self.p_o_up(n) ** 2

    def f(self, n):
        return self.A * self.y0(n) * self.p_o**2 / self.p_o_up(n)

    def g(self, n):
        return (self.A - 1) * self.y0(n) * self.p_o_up(n) / self.p_o
    
    def inv(self, n):
        return (self.bands_x[n] + self.f(n)) * (self.bands_y[n] + self.g(n))

    def inv_up(self, n):
        return self.p_o * self.A**2 * self.y0(n)**2
    
    def band_width(self, n):
        return self.p_o_up(n) / self.A
    
    def _y0(
            self,
            x, 
            y, 
            p_o, 
            p_o_up
        ):
        # solve quadratic:
        # p_o * A * y0**2 - y0 * (p_oracle_up/p_o * (A-1) * x + p_o**2/p_oracle_up * A * y) - xy = 0
        a = p_o * self.A
        b = p_o_up * (self.A-1) * x / p_o + p_o**2/p_o_up * self.A * y
        c = x * y
        return (b + (b**2 - 4*a*c)**0.5) / (2*a)
    
    def y0(self, n):
        """
        @notice wrapper to get _y0 for input band
        """
        return self._y0(self.bands_x[n], self.bands_y[n], self.p_o, self.p_o_up(n))

    def plot_reserves(self):
        """
        @notice Plot reserves in each band
        NOTE: for now, assume collateral price is = oracle price, and crvUSD price = $1
        """
        band_range = range(self.min_band, self.max_band)
        bands_x = [self.bands_x[i] for i in band_range]
        bands_y = [self.bands_y[i] * self.p_o for i in band_range]
        band_edges = [self.p_o_down(i) for i in band_range]
        band_widths = [self.band_width(i)*0.9 for i in band_range]

        plt.bar(band_edges, bands_y, color='darkblue', width=band_widths, label='Collateral')
        plt.bar(band_edges, bands_x, bottom=bands_y, color='darkred', width=band_widths, label='crvUSD')
        plt.xlabel('p_o_down[n] (USD)')
        plt.ylabel('Reserves (USD)')
        plt.title('LLAMMA Collateral Distribution')
        plt.xticks([round(i) for i in band_edges], rotation=45)
        plt.show()