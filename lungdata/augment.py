# str
from abc import ABC, abstractmethod
from typing import Sequence
from dataclasses import dataclass, astuple

# third
import numpy as np

# local
from .records import Record, record_stats


class Aug(ABC):
    """
    Augmentation interface
    example implentation of Aug for Linear
    >>> @dataclass
    >>> class Linear(Aug):
    >>>    k: float
    >>>    m: float
    >>>
    >>>    def modify(self, x):
    >>>        y = x * self.k + self.m
    >>>        return y
    """

    @property
    def name(self) -> str:
        return type(self).__name__ + "_".join(astuple(self))

    def __repr__(self):
        return self.name

    def __hash__(self):
        hash(self.name)

    @abstractmethod
    def modify(self, sound: np.ndarray) -> np.ndarray:
        """
        Modifier to be applied to the sound
        """
        pass


Augs = Sequence[Aug]


@dataclass
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
    speed: float

    def modify(self, x):
        xp = np.arange(len(x))
        xq = np.arange(0, len(x), self.speed)
        return np.interp(xq, xp, x)


rng = np.random.default_rng(123)


@dataclass
class Noise(Aug):
    magnitude: float = 0.005

    def modify(self, x):
        return x + self.magnitude * rng.standard_normal(x.shape)


pitch_augs = [Pitch(speed) for speed in np.linspace(0.7, 1.5, 5) if speed != 1.0]
noise_augs = [Noise(mag) for mag in (0.002, 0.004, 0.006)]
trunk_augs = [Trunc(f0, f1) for f1, f0 in ((0, 0.7), (0.3, 1))]


DEFAULT_AUGS = pitch_augs + noise_augs + trunk_augs + [Noop()]
