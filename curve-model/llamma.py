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

    def p_up(self, n):
        return self.base_price * (self.A/(self.A - 1))**n
    
    def create_bands(self):
        # Create 10 bands above and below the base price
        pass 