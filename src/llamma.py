from collections import defaultdict
import matplotlib.pyplot as plt

EPSILON = 1e-18 # to avoid division by 0
DEAD_SHARES = 1e-15 # to init shares in a band

plt.rcParams["font.family"] = "serif"
plt.rcParams.update({'font.size': 10})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

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
        self.user_shares = defaultdict(lambda: defaultdict(float))
        self.total_shares = defaultdict(float)
        self.active_band = 0
        self.min_band = 0
        self.max_band = 0

        # Set dependencies
        self.oracle = oracle

    def swap(self, amt_in, amt_out, tkn_in, tkn_out):
        """
        @notice This is a soft liquidation.
        """
        return

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

    def withdraw(self):
        pass

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
        return (x + self.f(n)) * (y + self.g(n))
    
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
        return (self.x + self.f(n)) * (self.y + self.g(n))

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
        return (-b + (b**2 - 4*a*c)**0.5) / (2*a)
    
    def y0(self, n):
        """
        @notice wrapper to get _y0 for input band
        """
        return self._y0(self.bands_x[n], self.bands_y[n], self.p_o, self.p_o_up(n))

    def plot_reserves(self):
        """
        @notice Plot reserves in each band
        NOTE: for now, assume collateral price is = AMM price, and crvUSD price = $1
        """
        band_range = range(self.min_band, self.max_band)
        bands_x = [self.bands_x[i] for i in band_range]
        bands_y = [self.bands_y[i] * self.p() for i in band_range]
        band_edges = [self.p_o_down(i) for i in band_range]
        band_widths = [self.band_width(i)*0.9 for i in band_range]

        plt.bar(band_edges, bands_y, color='darkblue', width=band_widths, label='Collateral')
        plt.bar(band_edges, bands_x, bottom=bands_y, color='darkred', width=band_widths, label='crvUSD')
        plt.xlabel('p_o_down[n] (USD)')
        plt.ylabel('Reserves (USD)')
        plt.title('LLAMMA Collateral Distribution')
        plt.xticks([round(i) for i in band_edges], rotation=45)
        plt.show()