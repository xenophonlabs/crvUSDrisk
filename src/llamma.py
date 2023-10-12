from collections import defaultdict

class LLAMMA:

    __slots__ = (

        # === Parameters === #
        'base_price' # Price at contract creation
        'A', # Amplification factor (width) for bands
        'fee', # Fee charged on swaps
        'admin_fee', # Pct of fees that go to admin
        'MAX_TICKS' # Define as an input param (instead of hard-coded at 50)

        # === State variables === #
        'bands_x', # bands_x[n] = stablecoins in band n
        'bands_y', # bands_y[n] = collateral in band n
        'user_shares', # user_shares[user, n] = user's share of band n
        'total_shares', # total_shares[n] = total shares in band n
        'active_band', # band for current price
        'min_band', # bands below this are empty
        'max_band', # bands above this are empty

        # === Dependencies/Inputs === #
        'oracle', # Oracle object
        'p_o', # Oracle price

        # === Helpers === #
        'p', # Current AMM price
        'p_c_up', # p_c_up[n] = Current max price of band n
        'p_c_down', # p_c_down[n] = Current min price of band n
        'p_o_up', # p_o_up[n] = Max price of band n if trades are adiabatic (p ~ p_o)
        'p_o_down', # p_o_down[n] = Min price of band n if trades are adiabatic (p ~ p_o)
        'f', # f[n] = A * y0 * p_o^2 / p_o_up[n]
        'g', # g[n] = (A - 1) * y0 * p_o_up[n] / p_o
        'inv', # inv[n] = (x + f[n])(y + g[n])
        'y0', # y0[n] = y reserves for band n if p = p_o = p_o_up[n]
        'inv_up', # inv_up[n] = p_o * A^2 * y0[n]^2
    )

    def __init__(
            self, 
            A, 
            base_price, 
            oracle,
            fee, 
            admin_fee,
        ):
        self.A = A
        self.base_price = base_price # TODO: eventually updated by interest rate
        self.fee = fee
        self.admin_fee = admin_fee

        self.bands_x = defaultdict(int)
        self.bands_y = defaultdict(int)
        self.shares = defaultdict(lambda: defaultdict(lambda: 0))

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

    def p_up(self, n):
        return self.base_price * (self.A/(self.A - 1))**n
    

    def create_bands(self):
        # Create 10 bands above and below the base price
        pass class LLAMMA:

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

    def p_up(self, n):
        return self.base_price * (self.A/(self.A - 1))**n
    

    def create_bands(self):
        # Create 10 bands above and below the base price
        pass 