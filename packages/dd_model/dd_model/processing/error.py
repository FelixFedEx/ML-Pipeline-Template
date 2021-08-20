from __future__ import absolute_import


class BaseError(Exception):
    """Base package error."""


class InvalidModelInputError(BaseError):
    """Model input contains an error."""
