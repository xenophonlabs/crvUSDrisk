from typing import Type
from sqlalchemy import (
    Column,
    Integer,
    Numeric,
    Float,
    String,
    ForeignKey,
    Table,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import as_declarative

@as_declarative()
class Base:
    __tablename__: str
    __table__: Table

Entity = Type[Base]

class Token(Base):
    __tablename__ = "tokens"
    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False)
    decimals = Column(Integer, nullable=False)


class Quote(Base):
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
