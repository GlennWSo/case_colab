# std
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Tuple

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
    def data_dict(self):
        return {key: val for key, val in asdict(self).items() if key != "sr"}

    def mean(self) -> SoundFeatures:
        data = asdict(self)
        for key, val in data.items():
            if key == "sr":
                continue
            data[key] = val.mean(axis=1)
        return type(self)(**data)

    def shapes(self) -> Tuple:
        return {
            key: val.shape
            for key, val in asdict(self).items()
            if isinstance(val, np.ndarray)
        }

    def __eq__(self, o: SoundFeatures) -> bool:
        if not isinstance(o, SoundFeatures):
            return False

        if not self.shapes() == o.shapes():
            return False

        f1 = self.data_dict.values()
        f2 = self.data_dict.values()
        data_eq = all((f1 == f2).all() for f1, f2 in zip(f1, f2))
        return data_eq and self.sr == o.sr

    def __repr__(self) -> str:
        return str(self.shapes())
