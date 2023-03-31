# str
from abc import ABC, abstractmethod
from typing import Sequence
from dataclasses import dataclass, astuple

# third
import numpy as np
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift


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

    @abstractmethod
    def __call__(self, sound: np.ndarray, sr: int) -> np.ndarray:
        """
        Modifier to be applied to the sound
        """
        pass


Augs = Sequence[Aug]


@dataclass()
class Noop(Aug):
    @staticmethod
    def __call__(x, _sr):
        return x

    def __hash__(self):
        """
        use dataclass values and cls for hasing
        """
        return hash((type(self), astuple(self)))


@dataclass
class Trunc(Aug):
    f_start: float = 0.0
    f_end: float = 1.0

    def __call__(self, x, _sr):
        s0 = int(len(x) * self.f_start)
        s1 = int(len(x) * self.f_end)
        return x[s0:s1]

    def __hash__(self):
        """
        use dataclass values and cls for hasing
        """
        return hash((type(self), astuple(self)))


@dataclass
class Pitch(Aug):
    speed: float

    def __call__(self, x, _sr):
        xp = np.arange(len(x))
        xq = np.arange(0, len(x), self.speed)
        return np.interp(xq, xp, x)

    def __hash__(self):
        """
        use dataclass values and cls for hasing
        """
        return hash((type(self), astuple(self)))


rng = np.random.default_rng(123)


@dataclass
class Noise(Aug):
    magnitude: float = 0.005

    def __call__(self, x, _sr):
        return x + self.magnitude * rng.standard_normal(x.shape)

    def __hash__(self):
        """
        use dataclass values and cls for hasing
        """
        return hash((type(self), astuple(self)))


pitch_augs = [Pitch(speed) for speed in np.linspace(0.7, 1.5, 5) if speed != 1.0]
noise_augs = [Noise(mag) for mag in (0.002, 0.004, 0.006)]
trunk_augs = [Trunc(f0, f1) for f1, f0 in ((0, 0.7), (0.3, 1))]


DEFAULT_AUGS = pitch_augs + noise_augs + trunk_augs + [Noop()]
