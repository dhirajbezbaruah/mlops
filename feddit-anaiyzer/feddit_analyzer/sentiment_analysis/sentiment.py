"""Enumeration of possible sentiment values."""

from enum import StrEnum
from typing import Self

from .errors import InvalidPolarityError

_POL_TH = 0


class Sentiment(StrEnum):
    """Enumeration of possible sentiment values."""

    POSITIVE = "positive"
    NEGATIVE = "negative"

    @classmethod
    def from_polarity(cls, polarity: float) -> Self:
        """Convert a polarity value to a sentiment value.

        The split threshold is 0, as polarity values are between -1 and 1.

        :param polarity: The polarity value to convert.
        :return: The corresponding sentiment value.
        """
        if not -1 <= polarity <= 1:
            raise InvalidPolarityError(polarity)

        return cls.POSITIVE if polarity >= _POL_TH else cls.NEGATIVE
