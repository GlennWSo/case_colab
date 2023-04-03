from __future__ import annotations
from time import time
from dataclasses import dataclass, fields
from typing import List, Sequence, Iterator, Callable, Dict, Tuple
import pickle
import toml
from functools import cached_property, cache

# third
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import librosa

# local
from .features import SoundFeatures
from .records import Record, record_stats
from .augment import Aug, Augs, DEFAULT_AUGS
from .utils import shallow_dict

rng = np.random.default_rng(1337)


@dataclass(unsafe_hash=True)
class DataPoint:
    # patient lookup
    record: Record
    aug: Aug
    features: SoundFeatures

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


def augment_sound(
    file: str, augs: Augs, map, **kwargs
) -> Dict[SoundFeatures, List[Aug]]:
    sound, sr = librosa.load(file)
    aug_sounds = augs.apply(sound, sr)
    return {
        aug: [SoundFeatures.from_sound(sound, sr, map, **kwargs) for sound in sounds]
        for aug, sounds in aug_sounds.items()
    }


def mk_augmented_points(
    r: Record | str, augs: Augs, map=None, **kwargs
) -> List[DataPoint]:
    """
    Create a sequance of points from single recording using augmentation
    """
    if isinstance(r, str):
        r = Record(r)

    aug_features = augment_sound(r.file, augs, map, **kwargs)

    data = []
    for aug, feature_list in aug_features.items():
        data.extend(DataPoint(r, aug, sf) for sf in feature_list)

    return data


class DataSet:
    def __init__(self, data: Sequence[DataPoint]):
        self.data: np.ndarray = np.array(data)
        self.encode()

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

    def __hash__(self):
        return hash(tuple(self.data))

    @cached_property
    def aug_dict(self) -> Dict[str, Aug]:
        """
        all aug types used in this dataset
        """
        augs = set(dp.aug for dp in self)
        return {aug.name: aug for aug in augs}

    @cached_property
    def unique_records(self) -> Tuple[Record]:
        return tuple(set(dp.record for dp in self))

    def aug_count(self, sum=False):
        """
        if sum:
            return total use of each aug
        else:
            use per record
        """
        count = {
            key: len(self.filter(lambda dp: dp.aug == aug))
            for key, aug in self.aug_dict.items()
        }
        if sum:
            return count
        for key in count:
            count[key] //= len(self.unique_records)
        return count

    def limit_augs(self, aug_limits: Dict[str, int]) -> DataSet:
        """
        filter self to limit the number uses for each augmentation on each record
        augs not mentationed will in caps will be unlimted
        """
        sub = self.filter(lambda dp: dp.aug.name not in set(aug_limits))

        limited_sub = self.filter(lambda dp: dp.aug.name in set(aug_limits))
        records = set(dp.record for dp in self)
        dps = []
        for r in records:
            r_data = limited_sub.filter(lambda dp: dp.record == r)
            caps = aug_limits.copy()
            for dp in r_data:
                if caps[dp.aug.name] > 0:
                    caps[dp.aug.name] -= 1
                    dps.append(dp)

        new_data = np.hstack((sub, np.array(dps)))
        return type(self)(new_data)

    @property
    def feature_names(self):
        return tuple(self[0].features.data.keys())

    @cache
    def get_features(self, feature_names: Sequence[str] = []):
        """
        concactinates data in features into a single array
        shape: [n_datapoints, (time_dim)]
        """
        assert self.is_time_homo(), """
        Datapoints must be homogeneous build arrays
            hint: try use method pad or map to make homologous data
        """
        if type(feature_names) == str:
            np.array([dp.features.data[feature_names] for dp in self])

        if len(feature_names) == 0:
            feature_names = self.feature_names
        return np.array(
            [
                np.concatenate(
                    [
                        values
                        for key, values in dp.features.data.items()
                        if key in feature_names
                    ]
                )
                for dp in self
            ]
        )

    @cached_property
    def features(self) -> np.ndarray:
        """
        get a concatenated array of all features.
            tip: use get_features when a subset of all features is required
        """
        return self.get_features()

    @property
    def records(self) -> List[Record]:
        return [dp.record for dp in self]

    @property
    def labels(self) -> pd.DataFrame:
        """
        diagnose, age, sex, loc, mode, equip
        """
        data = {
            "diagnose": [],
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
        filters the dataset in way that keeps datapoints with specified augmentations and balances the count of diagnoses

        if diag_size == 0:
            it will be set to the smalest diag class
        """
        if diag_size == 0:
            diag_size = min(val for val in self.stats["counts"]["diagnose"].values())

        warning = """
            warning: not enough data points of diagnose: {name}
                {diag_size} samples requested for each diagnosis,
                but dataset only has {n} of {name} (after aug_filter is applied)
        """
        diag_names = set(self.labels["diagnose"])
        sampled_dp: List[DataPoint] = []
        for name in diag_names:
            sub_set = self.filter(lambda dp: dp.record.diagnose == name)
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
    def load_wavs(
        cls, s=slice(0, -1), records=None, augs=None, map=None, **kwargs
    ) -> DataSet:
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
            data.extend(mk_augmented_points(r, augs, map, **kwargs))
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
        stats = record_stats(self.records)
        stats["augmentations"] = self.aug_count(sum=True)
        return stats

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
            pickle.dump(self.data.tolist(), file)

    def dumps(self) -> str:
        """
        stringy pickle
        """
        return pickle.dumps(self.data.tolist())

    @classmethod
    def loads(cls, dumps: str):
        """
        load cls from stringy pickle
        """
        return cls(pickle.loads(dumps))

    @classmethod
    def load_pickle(cls, path) -> DataSet:
        with open(path, mode="rb") as file:
            data = pickle.load(file)
        return cls(data)
