from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import scoped_session
from sqlalchemy.dialects.postgresql import insert
import pandas as pd
from .models import Base, Token, Quote
from typing import Union, List, Dict

from ..configs import config


class DataHandler:
    def __init__(self, uri=config.URI):
        self.uri = uri
        self.engine = create_engine(uri)
        self.session_factory = sessionmaker(bind=self.engine)
        self.session = scoped_session(self.session_factory)

    def close(self):
        self.session.remove()

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
            # print(row)
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
