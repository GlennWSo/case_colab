# std
from __future__ import annotations
import os

# third
import pandas as pd
import numpy as np
from typing import List, Callable, Dict, Sequence, Optional
from dataclasses import dataclass, field
import librosa

# local
from .path import META_PATH, SAMPLES_PATH
from .features import SoundFeatures


def str2float(s: str) -> float:
    """
    if parese fails returns float("nan")
    """
    try:
        f = float(s)
    except ValueError:
        f = float("nan")
    return f


def get_patient_data():
    def _mk_diag_data():
        diag_path = os.path.join(META_PATH, "ICBHI_Challenge_diagnosis.txt")
        with open(diag_path, "r") as file:
            lines = file.readlines()
        split_lines = (line.split() for line in lines)
        diag_map = {int(parts[0]): parts[1] for parts in split_lines if len(parts) == 2}
        return pd.Series(diag_map)

    def _mk_demo_data():
        demo_path = os.path.join(META_PATH, "demographic_info.txt")
        with open(demo_path, "r") as file:
            lines = file.readlines()
        split_lines = (line.split() for line in lines)
        demo_map = {
            int(parts[0]): {
                "age": str2float(parts[1]),
                "sex": parts[2],
            }
            for parts in split_lines
            if len(parts) > 2
        }
        return pd.DataFrame.from_dict(demo_map, orient="index")

    patient_data = _mk_demo_data()
    diag_data = _mk_diag_data()
    patient_data.insert(0, "diagnose", diag_data)

    return patient_data


PDATA = get_patient_data()

rare_limit = 3
all_diag_names = set(np.unique(PDATA["diagnose"]))
p_diag_count = {name: (PDATA["diagnose"] == name).sum() for name in all_diag_names}
rare_diags = {key for key, val in p_diag_count.items() if val <= rare_limit}
diag_names = all_diag_names - rare_diags


@dataclass()
class Record:
    """
    Contains all known meta/labels about records

    # abrivations:
    pid = patient id
    diag = diagnosis
    rid = record id # Note: (pid+rid) is a unique identifier for the sound files
    loc = location on the body where sound was taken
    mode = multi or single channel recording
    equip = stetoscope used
    """

    file: str = field(repr=False)

    pid: int = field(init=False)
    age: float = field(init=False)
    sex: str = field(init=False)
    diagnose: str = field(init=False)

    rid: str = field(init=False)
    loc: str = field(init=False)
    mode: str = field(init=False)
    equip: str = field(init=False)
    # lung cycles and anomalies
    annotations_file: str = field(init=False, repr=False)

    def __hash__(self) -> int:
        return hash(self.file)

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

    @staticmethod
    def limit_patient(pid: int, recs: Sequence[Record], caps: Dict) -> bool:
        """
        deterimine if that maximum allowed recordings per patient has been reached
        """
        diagnose = PDATA["diagnose"][pid]
        try:
            limit = caps[diagnose]
        except KeyError:
            return False

        return limit <= sum(1 for r in recs if r.pid == pid)

    default_caps = {
        "COPD": 4,
    }
    default_caps.update({name: 0 for name in rare_diags})

    @classmethod
    def load_wavs(
        cls, folder: str = SAMPLES_PATH, s=slice(-1), caps=default_caps
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


def record_stats(recs: Sequence[Record]):
    diag_names = set(r.diagnose for r in recs)
    counts = {
        "diagnose": {
            name: sum(1 for r in recs if r.diagnose == name) for name in diag_names
        },
        "gender": {
            name: sum(1 for r in recs if r.sex == name) for name in {"F", "M", "NA"}
        },
    }
    max_count = {name: max(counts[name].values()) for name in counts}

    major_fraction = {}

    for stat_type, max_n in max_count.items():
        major_fraction[stat_type] = {
            key: value / max_n for key, value in counts[stat_type].items()
        }

    stats = {
        "counts": counts,
        "major_fraction": major_fraction,
    }
    return stats
