from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    Float,
    String,
    ForeignKey,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()


class Token(Base):
    __tablename__ = "tokens"
    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    decimals = Column(Integer, nullable=False)


class Quote(Base):
    __tablename__ = "quotes"
    src = Column(String, ForeignKey("tokens.id"), primary_key=True)
    dst = Column(String, ForeignKey("tokens.id"), primary_key=True)
    in_amount = Column(BigInteger)
    out_amount = Column(BigInteger)
    gas = Column(Integer)
    price = Column(Float)
    protocols = Column(JSONB)
    timestamp = Column(Integer, primary_key=True)
