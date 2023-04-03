# std
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, Callable, Dict, List

# third
import numpy as np
import librosa


# local
from .augment import Augs


@dataclass()
class SoundFeatures:
    """
    # Data Description

    0. sr (Sample Rate)
    1. mfccs (Mel frequency cepstral coefficients)
    2. chroma
    3. mel Spectrogram
    4. tonnetz (Tonal Centroid Features)
    """

    sr: int
    mffcs: np.ndarray
    chroma: np.ndarray
    mel: np.ndarray
    tonnetz: np.ndarray

    def update(self, NewSoundFeatures) -> SoundFeatures:
        return NewSoundFeatures(asdict(self))

    @classmethod
    def from_sound(cls, sound, sr, map=None, **kwargs):
        stft = np.abs(librosa.stft(sound))
        chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        data = {
            "sr": sr,
            "mffcs": librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=40),
            "chroma": chroma,
            "mel": librosa.feature.melspectrogram(y=sound, sr=sr),
            "tonnetz": librosa.feature.tonnetz(y=sound, sr=sr, chroma=chroma),
        }
        self = cls(**data)
        if map:
            self.map(map, inplace=True, **kwargs)
        return self

    @property
    def data(self) -> Dict:
        return {key: val for key, val in asdict(self).items() if key != "sr"}

    @property
    def values(self) -> np.ndarray:
        return tuple(self.data.values())

    def __hash__(self):
        vstr = "".join(str(self.values)) + str(self.sr)
        return hash(vstr)

    def map(self, func: Callable, inplace=False, *args, **kwargs) -> SoundFeatures:
        """
        Create a new data by mapping data in self using func and *arg, **kwargs

        if inplace==True:
            updates self with new data
        else:
            returns new instance of Self
        """
        if func is None:
            return self

        new_data = {key: func(val, *args, **kwargs) for key, val in self.data.items()}
        if not inplace:
            return type(self)(sr=self.sr, **new_data)
        for key, val in new_data.items():
            setattr(self, key, val)

    def shapes(self) -> Tuple:
        return {key: val.shape for key, val in self.data.items()}

    def has_tdim(self) -> bool:
        # all features must be non scalar
        values = self.values
        if any(np.isscalar(data) for data in values):
            return False
        # all features must be atleast 2d
        if not all(data.ndim >= 2 for data in values):
            return False
        return True

    def is_time_homo(self) -> bool:
        """
        is size of time dimesion the same for all features
        """
        if not self.has_tdim():
            return False
        values = self.values
        if len(values) == 1:
            return True
        len1 = values[0].shape[1]
        return all(data.shape[1] == len1 for data in values[1:])

    def max_tdim(self) -> int:
        """
        the largest time dimension among features
        """
        assert self.has_tdim(), "all sound features must have a time dimension"
        return max(val.shape[1] for val in self.values)

    def pad_feature(
        self, feature: np.ndarray, tdim: int, pad_end: bool, **kwargs
    ) -> np.ndarray:
        delta = tdim - feature.shape[1]
        assert delta >= 0, "target size must be >= current size of features"
        if pad_end:
            padding = ((0, 0), (0, delta))
        else:
            padding = ((0, 0), (delta, 0))
        return np.pad(feature, pad_width=padding, **kwargs)

    def pad(self, tdim: int, pad_end=False, inplace=False, **kwargs):
        """
        pads time dim of each feature to size of arg: "tdim"
        """
        padded_data = {
            key: self.pad_feature(val, tdim, pad_end, **kwargs)
            for key, val in self.data.items()
        }

        if not inplace:
            return SoundFeatures(self.sr, **padded_data)
        for key, val in padded_data.items():
            setattr(self, key, val)

    def __repr__(self):
        def scalar_xor_shape(x):
            if np.isscalar(x):
                return x
            if len(x) <= 1:
                return x
            return x.shape

        return str({key: scalar_xor_shape(val) for key, val in self.data.items()})

    def __eq__(self, o: SoundFeatures) -> bool:
        if not isinstance(o, SoundFeatures):
            return False

        if not self.shapes() == o.shapes():
            return False

        f1 = self.data.values()
        f2 = self.data.values()

        data_eq = all((f1 == f2).all() for f1, f2 in zip(f1, f2))
        return data_eq and self.sr == o.sr
