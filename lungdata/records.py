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
from .path import DB_PATH, SAMPLES_PATH
from .features import SoundFeatures


def get_patient_data():
    def _mk_diag_data():
        diag_path = os.path.join(DB_PATH, "ICBHI_Challenge_diagnosis.txt")
        with open(diag_path, "r") as file:
            lines = file.readlines()
        split_lines = (line.split() for line in lines)
        diag_map = {int(parts[0]): parts[1] for parts in split_lines if len(parts) == 2}
        return pd.Series(diag_map)

    def _mk_demo_data():
        demo_path = os.path.join(DB_PATH, "demographic_info.txt")
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


@dataclass
class Record:
    """
    Contains all known meta/labels about records

    # feilds:
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

    def get_features(self, mod: Optional[Callable] = None) -> SoundFeatures:
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
        folder: str = SAMPLES_PATH,
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


RECORDS = Record.load_wavs()