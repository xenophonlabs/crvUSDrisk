# fetch fresh stable price data

import sys

sys.path.append("/Users/starklab/Documents/Code/Xenophon/Curve/crvUSDrisk/")
from src.utils.coinbaseprices import CoinbasePrices
path_to_data_dir = "/Users/starklab/Documents/Code/Xenophon/Curve/crvUSDrisk/data/"

cp = CoinbasePrices()

# cp.fetch_prices(asset_pair="USDT-USDC",gran=3600,initial_time = 1641013200,path_to_data_dir=path_to_data_dir)
cp.fetch_prices(asset_pair="ETH-USD",gran=86400,initial_time = 1641013200,path_to_data_dir=path_to_data_dir)
cp.fetch_prices(asset_pair="BTC-USD",gran=86400,initial_time = 1641013200,path_to_data_dir=path_to_data_dir)