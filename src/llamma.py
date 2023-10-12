from collections import defaultdict

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
        n0 = self.active_band
        assert n1 < n0
        pass

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
