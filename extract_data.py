# std
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict, astuple, fields
from functools import total_ordering
from time import time
from typing import List, Callable, Tuple, Dict, Sequence, Iterator
import os
import sys

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


PDATA = get_patient_data()
rare_limit = 3


def diag_count(data):
    diag_names = np.unique(data["diag"])
    return {name: (data["diag"] == name).sum() for name in diag_names}


rare_diags = {key for key, val in diag_count(PDATA).items() if val <= rare_limit}


# TODO
def diag_pie():
    """
    make pie plot of diags
    """
    pass
    # diag_types = set(patient_data["diag"])
    # print("All diag types:", diag_types)
    # print(patient_data.loc[101:106, :])


@dataclass
class SoundFeatures:
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
            "mel": librosa.feature.melspectrogram(sound, sr=sr),
            "tonnetz": librosa.feature.melspectrogram(sound, sr=sr),
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
        data = PDATA.loc[self.pid, :].to_dict()
        for key, value in data.items():
            setattr(self, key, value)

    def get_features(self, mod: Optional[SoundMod] = None) -> SoundFeatures:
        sound, sr = librosa.load(self.file)
        if mod is not None:
            sound = mod(sound)
        return SoundFeatures.from_sound(sound, sr)

    @staticmethod
    def limit_patient(pid: int, recs: Sequence[Record], caps: Dict) -> bool:
        """
        deterimine if that maximum allowed recordings per patient has been reached
        """
        diag = PDATA["diag"][pid]
        try:
            limit = caps[diag]
        except KeyError:
            return False

        return limit <= sum(1 for r in recs if r.pid == pid)

    @classmethod
    def load_wavs(
        cls,
        folder: str = samples_path,
        s=slice(0, -1),
        caps={"COPD": 4},
    ) -> List[Record]:
        wav_paths = [
            os.path.join(folder, file)
            for file in os.listdir(folder)
            if os.path.splitext(file)[1] == ".wav"
        ]
        recs = [cls(path) for path in wav_paths][s]
        if caps is None:
            return recs
        capped = []
        for r in recs:
            if cls.limit_patient(r.pid, capped, caps):
                continue
            capped.append(r)
        return capped


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
augs1 = [Noop] + speeds


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

    def reduce(self) -> DataPoint:
        mfeatures = self.features.mean()
        return type(self)(self.record, self.aug, mfeatures)

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


def s2hms(seconds: float, decimals: int = 1) -> str:
    dt = seconds
    hms = (
        str(dt // 3600),
        str((dt % 3600) // 60),
        str(round(dt % 60, decimals)),
    )
    return ":".join(hms)


class DataSet:
    def __init__(self, data: Sequence[DataPoint]):
        self.data: Sequence[DataPoint] = data

    def __getitem__(self, s):
        return self.data[s]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[DataPoint]:
        return iter(self.data)

    def __eq__(self, o: DataSet) -> bool:
        if len(self) != len(o):
            return False
        return all(p1 == p2 for p1, p2 in zip(self, o))

    @classmethod
    def load_wavs(cls, augs: Augs, folder: str = samples_path, n=-1) -> DataSet:
        records = Record.load_wavs(folder, slice(0, n))
        data = []
        n_recs = len(records)
        for r in records:
            print(r)
        print(f"extracting data from {n_recs} records...")
        dts = []
        for i, r in enumerate(records):
            t0 = time()
            data.extend(DataPoint.mk_augmented_points(r, augs))
            dt = time() - t0
            dts.append(dt)
            avg_dt = sum(dts) / len(dts)
            eta = s2hms(avg_dt * (n_recs - i))
            print(
                f"augmented datapoints: {len(data)}, processed records: {i}/{n_recs}, eta: {eta}"
            )
        return cls(data)

    def __repr__(self):
        lines = [f"{type(self).__name__}: {id(self)}\n---"]
        lines.extend(str(point) for point in self.data)
        return "\n\n".join(lines)

    def reduce(self) -> DataSet:
        data = [point.reduce() for point in self.data]
        return type(self)(data)

    def save_pickle(self, path: str):
        print(f"saving data at {path}")
        with open(path, mode="wb") as file:
            pickle.dump(self, file)

    @classmethod
    def load_pickle(cls, path) -> DataSet:
        with open(path, mode="rb") as file:
            instance = pickle.load(file)
        return instance


def cli_make_dataset():
    n = int(input("load n records\n"))
    print(f"loading {n} records")
    data = DataSet.load_wavs(augs1, n=n)
    print(data)
    print("---\n")

    path = input("save: [path/no]\n").strip()
    low = path.lower()
    if low == "no" or low == "n" or low == "":
        print("skipping save")
        return data
    data.save_pickle("test.data")
    return data


if __name__ == "__main__":
    cli_make_dataset()
