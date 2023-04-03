from __future__ import annotations

# str
from abc import ABC, abstractmethod
from typing import Sequence, Callable, List, Iterator, Tuple, Dict
from dataclasses import dataclass, astuple, is_dataclass, field

# third
import numpy as np
from audiomentations import (
    GainTransition,
    AddGaussianNoise,
    Mp3Compression,
    TimeStretch,
    PitchShift,
    Shift,
)


class Aug(ABC):
    """
    Augmentation interface
    example implentation of Aug for Linear
    # TODO
    """

    def func(self, x, sr) -> List[np.ndarray]:
        raise NotImplementedError

    def set_func(self, func):
        """
        a way to set a self.func for frozen instances
        """
        object.__setattr__(self, "func", func)

    def __call__(self, sound: np.ndarray, sr) -> List[np.ndarray]:
        return [self.func(sound, sr) for _ in range(self.n)]

    # todo record random parameters used during apply
    # @property
    # def rparams(self) -> Dict:
    # return self.func.parameters

    @property
    def fields_str(self) -> Tuple:
        if not is_dataclass(self):
            return ""
        return "_".join(str(e) for e in astuple(self))

    @property
    def default_name(self) -> str:
        """
        a default name
        """
        cname = type(self).__name__
        return f"{cname}_{self.field_str}"


@dataclass(frozen=True)
class RGainTransition(Aug):
    """
    random GainTransition

    parameters:
        min_gain(decibel)
        max_gain(decibel)
        min_t(seconds)
        max_t(seconds)
    """

    name: str
    n: int = 1
    # in decibel
    min_gain: float = -24.0
    max_gain: float = 6.0
    # duration in seconds
    min_t: float = 0.2
    max_t: float = 6.0

    def __post_init__(self):
        self.set_func(
            GainTransition(self.min_gain, self.max_gain, self.min_t, self.max_t, p=1.0)
        )


class Noop(Aug):
    """
    The noaugmentation augmentation. It just passes the data along
    """

    n = 1

    name = "noop"

    @staticmethod
    def func(x, _sr):
        return x


@dataclass
class RTrunc(Aug):
    name: str
    n: int = 1
    min_start: float = 0.0
    max_start: float = 0.3
    min_end: float = 7.0
    max_end: float = 1.0

    def __post_init__(self):
        assert self.min_start < self.max_start < self.min_end < self.max_end
        self.rng = np.random.default_rng()

    def randomize(self):
        self.f_start = self.rng.uniform(self.min_start, self.max_start)
        self.f_end = self.rng.uniform(self.min_end, self.max_end)

    def func(self, x, _sr):
        self.randomize()
        s0 = int(len(x) * self.f_start)
        s1 = int(len(x) * self.f_end)
        return x[s0:s1]

    def __hash__(self) -> int:
        return hash(str(self))


@dataclass(frozen=True)
class RPitch(Aug):
    """
    Random pitch shift the sound up or down without changing the tempo
    args:
    _shift: simitones -12..12
    """

    name: str
    n: int = 1
    min_shift: float = -4.0
    max_shift: float = 4.0

    def __post_init__(self):
        self.set_func(PitchShift(self.min_shift, self.max_shift, p=1.0))


@dataclass(frozen=True)
class RNoise(Aug):
    """
    random addgaussian noise
    """

    name: str
    n: int = 1
    min_amplitude: float = 0.001
    max_amplitude: float = 0.015

    def __post_init__(self):
        self.set_func(AddGaussianNoise(self.min_amplitude, self.max_amplitude, p=1.0))


@dataclass(frozen=True)
class RTimeStretch(Aug):
    """
    Random time strecthing
    """

    name: str
    n: int = 1
    min_rate: float = 0.75
    max_rate: float = 1.333

    def __post_init__(self):
        self.set_func(TimeStretch(self.min_rate, self.max_rate, p=1.0))


@dataclass
class Augs:
    augs: Sequence[Aug] = field(default_factory=list)

    def __post_init__(self):
        self.augs = tuple(self.augs)

    def __iter__(self) -> Iterator[Aug]:
        return iter(self.augs)

    def __getitem__(self, s):
        if type(s) == slice:
            return type(self)(self.data[s])
        if type(s) == list:
            return type(self)([self[i] for i in s])
        return self.data[s]

    def apply(self, x: np.ndarray, sr: int) -> Dict[Aug, List[np.ndarray]]:
        assert self.n > 0, "apply method needs self.augs to be non empty"
        return {aug: aug(x, sr) for aug in self}

    @property
    def n(self) -> int:
        return sum(aug.n for aug in self)

    def __hash__(self) -> int:
        return hash(self.augs)


noop = Noop()
rgain = RGainTransition("gain", 6)
rpitchup = RPitch("pitchup", 5, 1, 5)
rpitchdown = RPitch("pitchdown", 5, -5, -1)
rspeedup = RTimeStretch("speedup", 5, 1.1, 1.333)
rslowdown = RTimeStretch("slowdown", 5, 0.7, 0.9)
rnoise = RNoise("noise", 10, 0.001, 0.003)
rtrunc = RTrunc("truncate", 5, 0.1, 0.35, 0.65, 0.9)

DEFAULT_AUGS = Augs(
    (
        noop,
        rgain,
        rpitchup,
        rpitchdown,
        rspeedup,
        rslowdown,
        rnoise,
        rtrunc,
    )
)
