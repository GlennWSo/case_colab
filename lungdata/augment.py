# str
from abc import ABC, abstractmethod
from typing import Sequence
from dataclasses import dataclass

# third
import numpy as np

# local
from .records import Record


class Aug(ABC):
    """
    Augmentation rule
    """

    @abstractmethod
    def modify(self, sound: np.ndarray) -> np.ndarray:
        """
        Modifier to be applied to the sound
        """
        pass

    @staticmethod
    def apply2(_r: Record):
        """
        Rule for deteriming which records to apply this augmentation
        Default implementation: apply to all records
        """
        return True


Augs = Sequence[Aug]


class Noop(Aug):
    @staticmethod
    def modify(x):
        return x


@dataclass
class Trunc(Aug):
    f_start: float = 0.0
    f_end: float = 1.0

    def modify(self, x):
        s0 = int(len(x) * self.f_start)
        s1 = int(len(x) * self.f_end)
        return x[s0:s1]


@dataclass
class Pitch(Aug):
    factor: float

    @staticmethod
    def apply2(r: Record):
        return r.diag != "COPD"

    def modify(self, x):
        xp = np.arange(len(x))
        xq = np.arange(0, len(x), self.factor)
        return np.interp(xq, xp, x)


speeds = [Pitch(f) for f in np.linspace(0.7, 1.5, 5) if f != 1.0]
DEFUALT_AUGS = [Noop] + speeds
