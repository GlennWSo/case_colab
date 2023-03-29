# str
from abc import ABC, abstractmethod
from typing import Sequence, Dict
from dataclasses import dataclass

# third
import numpy as np

# local
from .records import Record, record_stats


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

    def apply2(_self, _r: Record) -> bool:
        """
        Rule for deteriming which records to apply this augmentation
        Default implementation: apply to all records
        """
        return True


Augs = Sequence[Aug]

# TODO serparate balancing and augmentation
# balancing should be handeld by dataset.py


@dataclass
class Balancer(Aug):
    """
    abstract class
    """

    seed: int
    diag_prob: Dict
    n: int

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def apply2(self, r) -> bool:
        try:
            p = 1 / self.diag_prob[r.diag] / self.n
            return self.rng.random() < p
        except KeyError:
            return True


@dataclass
class Noop(Balancer):
    @staticmethod
    def modify(x):
        return x


@dataclass
class Trunc(Balancer):
    f_start: float = 0.0
    f_end: float = 1.0

    def modify(self, x):
        s0 = int(len(x) * self.f_start)
        s1 = int(len(x) * self.f_end)
        return x[s0:s1]


@dataclass
class Pitch(Balancer):
    speed: float

    def modify(self, x):
        xp = np.arange(len(x))
        xq = np.arange(0, len(x), self.speed)
        return np.interp(xq, xp, x)


@dataclass
class Noise(Balancer):
    magnitude: float = 0.005

    def modify(self, x):
        return x + self.magnitude * np.random.randn(*x.shape)


def mk_balanced_augs(recs: Sequence[Record]) -> Augs:
    """
    makes a set of augmentations that balances diagionis
    """
    diag_fractions = record_stats(recs)["major_fraction"]["diag"]
    speeds = [x for x in np.linspace(0.7, 1.5, 10) if x != 1.0]
    noise_levels = [0.002 * 1.1**n for n in range(1, 11)]  # TODO 3 noise levels
    n_augs = len(speeds) + len(noise_levels) + 1

    pitch_augs = [
        Pitch(i, diag_fractions, n_augs, speed) for i, speed in enumerate(speeds)
    ]

    noise_augs = [
        Noise(i + len(pitch_augs), diag_fractions, n_augs, magnitude)
        for i, magnitude in enumerate(noise_levels)
    ]

    augs = [Noop(1337, diag_fractions, n_augs)] + pitch_augs + noise_augs
    return augs


DEFAULT_AUGS = [Noop(0, {}, 1)]
