from __future__ import annotations
from time import time
from dataclasses import dataclass, fields, asdict
from typing import List, Sequence, Iterator, Callable, Dict, Optional
import pickle
import toml
from functools import cached_property

# third
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# local
from .features import SoundFeatures
from .records import Record, record_stats
from .augment import Aug, Augs, DEFAULT_AUGS
from .utils import shallow_dict

rng = np.random.default_rng(1337)


@dataclass
class DataPoint:
    # patient lookup
    record: Record
    aug: Aug
    features: SoundFeatures

    def update(self, NewDataPoint=None, NewSoundFeatures=None) -> DataPoint:
        if NewSoundFeatures:
            self.features = NewSoundFeatures(**shallow_dict(self.features))

        if NewDataPoint:
            return NewDataPoint(**shallow_dict(self))

        return self

    @classmethod
    def mk_augmented_points(cls, record: Record | str, augs: Augs) -> List[DataPoint]:
        """
        Create a sequance of points from single recording using augmentation
        """
        if isinstance(record, Record):
            r = record
        else:
            r = Record(record)

        data = [cls(r, aug, r.get_features(aug.modify)) for aug in augs]

        return data

    def map(self, func, *args, inplace=False, **kwargs) -> DataPoint:
        """
        Create a new data by mapping data in self using func and *arg, **kwargs

        if inplace==True:
            updates self with new data
        else:
            returns new instance of Self
        """
        mfeatures = self.features.map(func, *args, **kwargs, inplace=inplace)
        if not inplace:
            return type(self)(self.record, self.aug, mfeatures)

    def pad(self, tdim: int, pad_end=False, inplace=False, **kwargs):
        pfeatures = self.features.pad(tdim, pad_end, inplace, **kwargs)
        if not inplace:
            return type(self)(self.record, self.aug, pfeatures)

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
        str(int(dt) // 3600),
        str((int(dt) % 3600) // 60),
        str(round(dt % 60, decimals)),
    )
    return ":".join(hms)


class DummyEncoder(LabelEncoder):
    def fit(self, y):
        pass

    def transform(self, y):
        return y

    def inverse_transform(self, y):
        return y


# augment frist without balancing.. Just increase the total n datapoints
# then let the students balnace and pick the augmentations from prepared data


class DataSet:
    def __init__(self, data: Sequence[DataPoint]):
        self.data = np.array(data)
        self.encode()

    def update(self, NewDataSet, NewDataPoint, NewSoundFeatures) -> DataSet:
        if NewSoundFeatures:
            self.data = [dp.update(NewDataPoint, NewSoundFeatures) for dp in self]

        if NewDataSet:
            return NewDataSet(self.data)
        return self

    def __getitem__(self, s):
        if type(s) == slice:
            return type(self)(self.data[s])
        if type(s) == list:
            return type(self)([self[i] for i in s])
        return self.data[s]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[DataPoint]:
        return iter(self.data)

    def __eq__(self, o: DataSet) -> bool:
        if len(self) != len(o):
            return False
        return all(p1 == p2 for p1, p2 in zip(self, o))

    @cached_property
    def features(self) -> np.ndarray:
        return pd.DataFrame(
            [dp.features.values for dp in self],
            columns=[key for key in self[0].features.data.keys()],
        )

    @property
    def records(self) -> List[Record]:
        return [dp.record for dp in self]

    @property
    def labels(self) -> pd.DataFrame:
        """
        diag, age, sex, loc, mode, equip
        """
        data = {
            "diag": [],
            "age": [],
            "sex": [],
            "loc": [],
            "mode": [],
            "equip": [],
        }

        for r in self.records:
            for key in data:
                data[key].append(getattr(r, key))

        for key, val in data.items():
            if key == "age":
                dtype = float
            else:
                dtype = str
            data[key] = pd.Series(val, name=key, dtype=dtype)

        return pd.DataFrame(data)

    @staticmethod
    def _encode_obj(data: pd.Series) -> LabelEncoder:
        """
        Create encoder for obj labels,
        for numeric labels, create dummy encoder that just pass along data
        """
        if data.dtype is not np.dtype("O"):
            return DummyEncoder()
        le = LabelEncoder()

        return le.fit(data)

    def encode(self) -> Dict["str", LabelEncoder]:
        """
        make an encoder for self.labels
        """
        self._encoder = {
            col_name: self._encode_obj(col_data)
            for col_name, col_data in self.labels.items()
        }
        return self._encoder

    @property
    def encoders(self) -> Dict["str", LabelEncoder]:
        if self._encoder is None:
            self.encode()
        return self._encoder

    def encoded_labels(self) -> pd.DataFrame:
        labels = self.labels
        coded = {
            name: self.encoders[name].transform(data) for name, data in labels.items()
        }
        return pd.DataFrame(coded)

    def filter(self, dp_filter: Callable[[DataPoint], bool]) -> DataSet:
        inds = [i for i, dp in enumerate(self) if dp_filter(dp)]
        return self[inds]

    def rpick(self, n) -> DataSet:
        """
        at random pick n datapoints from self and construct new DataSet
        """
        rand_inds = rng.choice(len(self), n, replace=False)
        rdata = [self[i] for i in rand_inds]
        return type(self)(rdata)

    def under_sample(self, diag_size: int = 0, aug_filter=None) -> DataSet:
        """
        filters the dataset in way that keeps datapoints with specified augmentations and balances the count of diags

        if diag_size == 0:
            it will be set to the smalest diag class
        """
        if diag_size == 0:
            diag_size = min(val for val in self.stats["counts"]["diag"].values())

        warning = """
            warning: not enough data points of diag: {name}
                {diag_size} samples requested for each diagnosis,
                but dataset only has {n} of {name} (after aug_filter is applied)
        """
        diag_names = set(self.labels["diag"])
        sampled_dp: List[DataPoint] = []
        for name in diag_names:
            sub_set = self.filter(lambda dp: dp.record.diag == name)
            if aug_filter is not None:
                sub_set = sub_set.filter(lambda dp: aug_filter(dp.aug))
            if len(sub_set) < diag_size:
                print(warning.format(name=name, diag_size=diag_size, n=len(sub_set)))
                sampled_dp.extend(sub_set.data)
                continue
            recs = set(dp.record for dp in sub_set)
            n_picks = 0
            n_consider = 0
            for r in recs:
                augsubset = sub_set.filter(lambda dp: dp.record == r)
                n_consider += len(augsubset)
                # n_picks/n_consider =aprox= diag_size/len(sub_set)
                n_ideal = n_consider * diag_size / len(sub_set)
                n = round(n_ideal - n_picks)
                n_picks += n
                picks = augsubset.rpick(n)
                sampled_dp.extend(picks)

        return type(self)(sampled_dp)

    def map(self, func: Callable, *args, inplace=False, **kwargs) -> DataSet:
        """
        Create new SoundFeatures by appling func.
        if inplace:
            update inplace
        else:
            return new DataSet
        """
        if not inplace:
            return type(self)([dp.map(func, *args, **kwargs) for dp in self])
        for dp in self:
            dp.map(func, *args, **kwargs, inplace=True)

    def has_time_dim(self) -> bool:
        if len(self) == 0:
            return False
        return all(dp.features.has_tdim() for dp in self)

    def max_tdim(self) -> int:
        return max(dp.features.max_tdim() for dp in self)

    def pad(self, pad_end=False, inplace=False, **kwargs):
        """
        Pad the soundfeatures so the time dimension size the same size for all datapoints
        if inplace:
            update inplace
        else:
            return a new instance of DataSet
        """
        tdim = self.max_tdim()
        if not inplace:
            return type(self)(
                [dp.pad(tdim, pad_end, inplace=False, **kwargs) for dp in self]
            )
        for dp in self:
            dp.pad(tdim, pad_end, inplace=True, **kwargs)

    def is_time_homo(self) -> bool:
        """
        checks if all features have same time size
        """
        if not self.has_time_dim():
            return False

        if len(self) == 1:
            return True

        len1 = self[0].features.max_tdim()
        for dp in self[1:]:
            if not dp.features.is_time_homo() or (dp.features.max_tdim() != len1):
                return False
        return True

    @classmethod
    def load_wavs(cls, records=None, augs=None, s=slice(0, -1)) -> DataSet:
        if records is None:
            records = Record.load_wavs()

        if type(s) == int:
            recs = [records[s]]
        else:
            recs = records[s]

        if augs is None:
            augs = DEFAULT_AUGS

        n_recs = len(recs)
        for r in recs:
            print(r)
        print(f"extracting data from {n_recs} records...")
        dts = []
        data = []
        for i, r in enumerate(recs):
            t0 = time()
            data.extend(DataPoint.mk_augmented_points(r, augs))
            dt = time() - t0
            dts.append(dt)
            avg_dt = sum(dts) / len(dts)
            eta = s2hms(avg_dt * (n_recs - i))
            print(
                f"augmented datapoints: {len(data)}, processed records: {i+1}/{n_recs}, eta: {eta}"
            )
        total_time = sum(dts)

        print(f"data compiling took: {s2hms(total_time)} ")
        return cls(data)

    @property
    def stats(self) -> Dict:
        return record_stats(self.records)

    @property
    def pretty_stats(self) -> str:
        return toml.dumps(self.stats)

    def __repr__(self) -> str:
        lines = [f"{type(self).__name__}: {id(self)}\n---"]
        lines.extend(str(point) for point in self.data)
        return "\n\n".join(lines)

    def __str__(self) -> str:
        """
        create a summary of self
        """
        msg = f"""
{type(self).__name__} at {id(self)}
len: {len(self)}

#stats
{self.pretty_stats}
        """
        return msg

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
