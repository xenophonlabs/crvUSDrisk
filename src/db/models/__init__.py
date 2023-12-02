"""Package for SQLAlchemy models for accessing our PG database."""
from sqlalchemy import (
    Column,
    Integer,
    Numeric,
    Float,
    String,
    ForeignKey,
    Table,
    MetaData,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import as_declarative


# pylint: disable=too-few-public-methods
@as_declarative()
class Base:
    """Base class for SQLAlchemy models."""

    __table__: Table
    metadata: MetaData


# pylint: disable=too-few-public-methods
class Token(Base):
    """A dimension table storing token metadata."""

    __tablename__ = "tokens"
    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    decimals = Column(Integer, nullable=False)


# pylint: disable=too-few-public-methods
class Quote(Base):
    """A fact table storing 1inch quotes."""

    __tablename__ = "quotes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    src = Column(String, ForeignKey("tokens.id"))
    dst = Column(String, ForeignKey("tokens.id"))
    in_amount = Column(Numeric)
    out_amount = Column(Numeric)
    gas = Column(Integer)
    price = Column(Float)
    protocols = Column(JSONB)
    timestamp = Column(Integer)
