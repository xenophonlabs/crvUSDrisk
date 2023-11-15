import pandas as pd
from src.db.datahandler import DataHandler

dh = DataHandler()
df = pd.read_csv("./data/1inch/tmp.csv")
dh.insert_quotes(df)
