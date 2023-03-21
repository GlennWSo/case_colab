# std
from __future__ import annotations
from abc import ABC, abstractmethod
import os
from typing import List, Callable, Tuple, Dict, Sequence
from dataclasses import dataclass, field, asdict, astuple, fields
from functools import total_ordering

# third
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
import pickle


db_path = os.environ["DB_PATH"]
samples_path = os.path.join(db_path, "samples")


def get_patient_data():
    def _mk_diag_data():
        diag_path = os.path.join(db_path, "ICBHI_Challenge_diagnosis.txt")
        with open(diag_path, "r") as file:
            lines = file.readlines()
        split_lines = (line.split() for line in lines)
        diag_map = {int(parts[0]): parts[1] for parts in split_lines if len(parts) == 2}
        return pd.Series(diag_map)

    def _mk_demo_data():
        demo_path = os.path.join(db_path, "demographic_info.txt")
        with open(demo_path, "r") as file:
            lines = file.readlines()
        split_lines = (line.split() for line in lines)
        demo_map = {
            int(parts[0]): {
                "age": parts[1],
                "sex": parts[2],
            }
            for parts in split_lines
            if len(parts) > 2
        }
        return pd.DataFrame.from_dict(demo_map, orient="index")

    patient_data = _mk_demo_data()
    diag_data = _mk_diag_data()
    patient_data.insert(0, "diag", diag_data)
    return patient_data


patient_data = get_patient_data()

diag_types = set(patient_data["diag"])
print("All diag types:", diag_types)

print(patient_data.loc[101:106, :])


@dataclass
class SoundFeatures:
    sr: int
    mffcs: np.ndarray
    chroma: np.ndarray
    mel: np.ndarray
    tonnetz: np.ndarray

    def __init__(self, sound: np.ndarray, sr: int):
        self.sr = sr
        stft = np.abs(librosa.stft(sound))
        self.mffcs = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=40)
        self.chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
        self.mel = librosa.feature.melspectrogram(sound, sr=sr)
        self.tonnetz = librosa.feature.melspectrogram(sound, sr=sr)

    def mean_concat(self):
        all_features = ()
        means = [np.mean(item, axis=1) for item in astuple(self)]
        return np.concatenate(means)

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

        f1 = astuple(self)
        f2 = astuple(o)
        return all((f1 == f2).all() for f1, f2 in zip(f1, f2))

    def __str__(self) -> str:
        return str(self.shapes())


SoundMod = Callable[[np.ndarray], np.ndarray]


@dataclass
class Record:
    """
    Contains all known meta/labels about records

    # abreviations:
    pid = patient id
    rid = record id # Note: (pid+rid) is a unique identifier for the sound files
    loc = location on the body where sound was taken
    mode = multi or single channel recording
    equip = stetoscope used
    """

    file: str = field(repr=False)

    pid: int = field(init=False)
    age: float = field(init=False)
    sex: str = field(init=False)
    diag: str = field(init=False)

    rid: str = field(init=False)
    loc: str = field(init=False)
    mode: str = field(init=False)
    equip: str = field(init=False)
    # lung cycles and anomalies
    annotations_file: str = field(init=False, repr=False)

    def __post_init__(self):
        """
        unpack the info in the file naming convention

        # example
        path = parrent/101_1b1_Al_sc_Meditron.wav
        meta data is delimeted with "_" like the fallowing order
        "pid", "rid", "loc", "mode", "equip"
        """
        root, ending = os.path.splitext(self.file)
        self.annotations_file = root + ".txt"
        base = os.path.basename(root)
        pid, self.rid, self.loc, self.mode, self.equip = base.split("_")
        self.pid = int(pid)
        data = patient_data.loc[self.pid, :].to_dict()
        for key, value in data.items():
            setattr(self, key, value)

    def get_features(self, mod: SoundMod) -> SoundFeatures:
        sound, sr = librosa.load(self.file)
        return SoundFeatures(mod(sound), sr)

    @classmethod
    def load_wavs(cls, folder: str = samples_path, n=-1) -> List[Record]:
        wav_paths = [
            os.path.join(folder, file)
            for file in os.listdir(folder)
            if os.path.splitext(file)[1] == ".wav"
        ]
        return [cls(path) for path in wav_paths][:n]


def shallow_dict(obj) -> Dict:
    """
    shallow converstion of a dataclass to dict
    """
    return dict((field.name, getattr(obj, field.name)) for field in fields(obj))


class Aug(ABC):
    """
    Augmentation
    """

    @abstractmethod
    def modify(self, data: np.ndarray) -> np.ndarray:
        pass

    @staticmethod
    def apply2(_r: Record):
        """
        Applies to all
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


trunc1 = Trunc(0, 0.9)
trunc2 = Trunc(0.2)


@dataclass
class Pitch(Aug):
    factor: float

    @staticmethod
    def apply2(r: Record):
        return bool(r.pid % 2)

    def modify(self, x):
        xp = np.arange(len(x))
        xq = np.arange(0, len(x), self.factor)
        return np.interp(xq, xp, x)


speedup1 = Pitch(1.1)
augs1 = [Noop, speedup1]


@dataclass
class DataPoint:
    # patient lookup
    record: Record
    aug: Aug
    features: SoundFeatures

    @classmethod
    def mk_augmented_points(cls, record: Record | str, augs: Augs) -> List[DataPoint]:
        """
        Create a sequance of points from single recording using augmentation
        """
        if isinstance(record, Record):
            r = record
        else:
            r = Record(record)

        data = [
            cls(r, aug, r.get_features(aug.modify)) for aug in augs if aug.apply2(r)
        ]

        return data

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}: {id(self)}"]
        lines.extend(f"{f.name}: {getattr(self, f.name)}" for f in fields(self))
        return "\n\t".join(lines)

    def save(self, path: str):
        with open(path, mode="wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path) -> DataPoint:
        with open(path, mode="rb") as file:
            return pickle.load(file)


class DataSet:
    def __init__(self, data: Sequence[DataPoint]):
        self.data = data

    def __get__(self, s):
        return self.data[s]

    @classmethod
    def load_wavs(cls, augs: Augs, folder: str = samples_path, n=-1) -> DataSet:
        records = Record.load_wavs(folder, n)
        data = []
        for r in records:
            data.extend(DataPoint.mk_augmented_points(r, augs))
        return cls(data)

    def __str__(self):
        return "\n\n".join(str(item) for item in self.data)

    def save(self, path: str):
        with open(path, mode="wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path) -> DataSet:
        with open(path, mode="rb") as file:
            return pickle.load(file)


# data = DataPoint.mk_augmented_points(soundfile1, augs)

data = DataSet.load_wavs(augs1, n=4)
print(data)

# data = DataSet.load_db([Noop])
# print(data)
# data.save("test.data")
# dataloaded = DataSet.load("test.data")
# print(dataloaded)
