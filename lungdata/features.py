# std
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple, Callable

# third
import numpy as np

# import soundfile as sf
import librosa


@dataclass
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

    @classmethod
    def from_sound(cls, sound, sr):
        stft = np.abs(librosa.stft(sound))
        kwargs = {
            "sr": sr,
            "mffcs": librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=40),
            "chroma": librosa.feature.chroma_stft(S=stft, sr=sr),
            "mel": librosa.feature.melspectrogram(y=sound, sr=sr),
            "tonnetz": librosa.feature.melspectrogram(y=sound, sr=sr),
        }
        return cls(**kwargs)

    @property
    def data(self):
        return {key: val for key, val in asdict(self).items() if key != "sr"}

    def map_inplace(self, func: Callable, *args, **kwargs):
        """
        updates data in self using func with values:
            found in self.data and in *args **kwargs.
        """
        for key, val in self.data.items():
            self.data[key] = func(val, *args, **kwargs)

    def map(self, func: Callable, *args, **kwargs) -> SoundFeatures:
        """
        Create a new instace of Self, by mapping self.data using func and *arg, **kwargs
        """
        new_data = {key: func(val, *args, **kwargs) for key, val in self.data.items()}
        return type(self)(sr=self.sr, **new_data)

    def shapes(self) -> Tuple:
        return {key: val.shape for key, val in self.data.items()}

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
