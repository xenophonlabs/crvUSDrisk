"""
Provides a `DataHandler` class 
for accessing our PG database.
"""
import logging
from typing import List, Type
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from sqlalchemy.dialects.postgresql import insert
from .models import Base, Token, Quote
from ..configs import URI, TOKEN_DTOs


class DataHandler:
    """
    The DataHandler class provides an interface
    for accessing our PG database. It provides
    methods for creating the database, inserting
    data, and querying data.
    """

    def __init__(self, uri=URI):
        self.uri = uri
        self.engine = create_engine(uri)
        self.session_factory = sessionmaker(bind=self.engine)
        self.session = scoped_session(self.session_factory)

    def close(self):
        """Close the SQLAlchemy session."""
        self.session.remove()

    def __enter__(self):
        """Enter DataHandler context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit DataHandler context manager."""
        self.close()

    def create_database(self):
        """Create SQL tables if they don't exist"""
        logging.info("Creating database...")
        Base.metadata.create_all(self.engine)
        logging.info("Inserting tokens from config...")
        self.insert_tokens()
        logging.info("Done.")

    def insert_tokens(self):
        """Create the tokens table based on the config."""
        tokens = [
            {"id": v["address"], "symbol": v["symbol"], "decimals": v["decimals"]}
            for v in TOKEN_DTOs.values()
        ]
        for token in tokens:
            self.session.add(Token(**token))
        self.session.commit()

    def insert_quotes(self, quotes: pd.DataFrame):
        """Given a dataframe of quotes, insert them sequentially in db."""
        if quotes.empty:
            return
        self.insert_df(quotes, Quote)

    def insert_df(
        self,
        df: pd.DataFrame,
        entity: Type[Base],
        replace: bool = False,
        index_elements: List[str] | None = None,
    ):
        """
        Insert rows from dataframe in `entity`. If `replace`
        is True, then conflicting rows will be updated.
        """
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
        entity: Type[Base],
        replace: bool = False,
        index_elements: List[str] | None = None,
    ):
        """
        Insert a list of dicts in `entity`. If `replace`
        is True, then conflicting rows will be updated.
        """
        if not lst:
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

    def process_quotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs the following processing steps:
            1. Create datetime index (floored to hours).
            2. Add reference prices for each 'round' of quotes:
                a. Quotes are fetched every hour. This is a 'round'.
                b. The reference price is the best price for
                that 'round'.
            3. Add price impact as the pct difference
            between the reference price and the quoted price.
        """
        df["in_amount"] = df["in_amount"].astype(float)
        df["out_amount"] = df["out_amount"].astype(float)

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

    def get_tokens(self, cols: List[str] | None = None) -> pd.DataFrame:
        """Get tokens from database."""
        if not cols:
            cols = ["id", "symbol", "decimals"]
        query = self.session.query(*[getattr(Token, col) for col in cols])
        results = query.all()
        if not results:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(results)

    # pylint: disable=too-many-arguments
    def get_quotes(
        self,
        pair: tuple | None = None,
        start: int | None = None,
        end: int | None = None,
        cols: List[str] | None = None,
        process: bool = False,
    ) -> pd.DataFrame:
        """
        Get 1inch quotes from database. Filter
        query by the input parameters.
        """
        if not cols:
            cols = [
                "src",
                "dst",
                "in_amount",
                "out_amount",
                "price",
                "timestamp",
            ]
        query = self.session.query(*[getattr(Quote, col) for col in cols])
        if pair:
            query = query.filter(Quote.src == pair[0], Quote.dst == pair[1])
        if start:
            query = query.filter(Quote.timestamp >= start)
        if end:
            query = query.filter(Quote.timestamp < end)
        results = query.all()
        if not results:
            return pd.DataFrame()
        results = pd.DataFrame.from_dict(results)
        if process:
            results = self.process_quotes(results)
        return results
