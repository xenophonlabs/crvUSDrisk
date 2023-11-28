from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from .models import Base, Token, Quote
from typing import List

from ..configs import config


class DataHandler:
    def __init__(self, uri=config.URI):
        self.uri = uri
        self.engine = create_engine(uri)
        self.session_factory = sessionmaker(bind=self.engine)
        self.session = scoped_session(self.session_factory)

    def close(self):
        self.session.remove()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def create_database(self):
        """Create tables if they don't exist"""
        print("Creating database...")
        Base.metadata.create_all(self.engine)
        print("Inserting tokens from config...")
        self.insert_tokens()
        print("Done.")

    def insert_tokens(self):
        """Create the tokens table based on the config."""
        tokens = [
            {"id": v["address"], "symbol": k, "decimals": v["decimals"]}
            for k, v in config.ALL.items()
        ]
        for token in tokens:
            self.session.add(Token(**token))
        self.session.commit()

    def insert_quotes(self, quotes: pd.DataFrame):
        """Given a dataframe of quotes, insert them sequentially in db."""
        if not len(quotes):
            return
        self.insert_df(quotes, Quote)

    def insert_df(
        self,
        df: pd.DataFrame,
        entity: object,
        replace: bool = False,
        index_elements: List[str] = [],
    ):
        """Insert rows from dataframe."""
        for _, row in df.iterrows():
            stmt = insert(entity.__table__).values(**row)
            if replace:
                stmt = stmt.on_conflict_do_update(
                    index_elements=index_elements, set_=dict(row)
                )
            else:
                stmt = stmt.on_conflict_do_nothing()
            try:
                self.session.execute(stmt)
            except Exception as e:
                self.session.rollback()
                raise e

        self.session.commit()

    def insert_list(
        self,
        lst: List[dict],
        entity: object,
        replace: bool = False,
        index_elements: List[str] = [],
    ):
        """Insert a list of dictionaries"""
        if len(lst) == 0:
            return

        for d in lst:
            stmt = insert(entity.__table__).values(**d)
            if replace:
                stmt = stmt.on_conflict_do_update(
                    index_elements=index_elements, set_=dict(d)
                )
            else:
                stmt = stmt.on_conflict_do_nothing()
            try:
                self.session.execute(stmt)
            except Exception as e:
                self.session.rollback()
                raise e

        self.session.commit()

    def process_quotes(
        self, quotes: pd.DataFrame, tokens: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Do the following:
            1. Convert amounts to human readable format.
            2. Create datetime index (floored to hours).
            3. Add reference prices for each block of quotes:
                a. Quotes are fetched every hour.
                b. The reference price is the best price for
                that block.
            4. Add price impact as the pct difference
            between the reference price and the quoted price.
        """
        # Fix decimals
        df = (
            pd.merge(quotes, tokens, left_on="src", right_on="id")
            .drop(["id", "symbol"], axis=1)
            .rename(columns={"decimals": "src_decimals"})
        )
        df = (
            pd.merge(df, tokens, left_on="dst", right_on="id")
            .drop(["id", "symbol"], axis=1)
            .rename(columns={"decimals": "dst_decimals"})
        )
        # TODO don't adjust for decimals and train EM 
        # on raw amounts
        df["in_amount"] = (df["in_amount"] / 10 ** df["src_decimals"]).astype(float)
        df["out_amount"] = (df["out_amount"] / 10 ** df["dst_decimals"]).astype(float)

        # Create datetime index floored to hours.
        df["hour"] = pd.to_datetime(df["timestamp"], unit="s").dt.floor("h")

        # Add reference price
        # TODO improve the way we get reference price.
        grouped = (
            df.groupby(["hour", "src", "dst"])
            .agg(reference_price=("price", "max"))
            .reset_index()
        )
        df = pd.merge(
            df,
            grouped,
            left_on=["hour", "src", "dst"],
            right_on=["hour", "src", "dst"],
        )

        # Add price impact
        df["price_impact"] = (df["reference_price"] - df["price"]) / df[
            "reference_price"
        ]

        df.set_index(["src", "dst"], inplace=True)
        df.sort_index(inplace=True)

        return df

    def get_tokens(self, cols: List[str] = ["id", "symbol", "decimals"]) -> dict:
        """Get tokens from database."""
        query = self.session.query(*[getattr(Token, col) for col in cols])
        results = query.all()
        if not len(results):
            return pd.DataFrame()
        return pd.DataFrame.from_dict(results)

    def get_quotes(
        self,
        pair: tuple = None,
        start: int = None,
        end: int = None,
        cols: List[str] = [
            "src",
            "dst",
            "in_amount",
            "out_amount",
            "price",
            "timestamp",
        ],
        process: bool = False,
    ) -> pd.DataFrame:
        """
        Get 1inch quotes from database. Filter
        query by the input parameters.
        """
        query = self.session.query(*[getattr(Quote, col) for col in cols])
        if pair:
            query = query.filter(Quote.src == pair[0], Quote.dst == pair[1])
        if start:
            query = query.filter(Quote.timestamp >= start)
        if end:
            query = query.filter(Quote.timestamp < end)
        results = query.all()
        if not len(results):
            return pd.DataFrame()
        results = pd.DataFrame.from_dict(results)
        if process:
            results = self.process_quotes(results, self.get_tokens())
        return results
