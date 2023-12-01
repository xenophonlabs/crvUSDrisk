"""Provides custom exceptions."""


class crvusdRiskException(Exception):
    """Base class for exceptions in this module."""


class coingeckoRateLimitException(crvusdRiskException):
    """Coingecko API Rate Limit Exceeded."""


class ccxtInvalidSymbolException(crvusdRiskException):
    """Exchange does not support input symbol."""
