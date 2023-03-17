import numpy as np
import os
import soundfile as sf


db_path = os.environ["DB_PATH"]


def get_diagnoisis_map():
    diag_path = os.path.join(db_path, "ICBHI_Challenge_diagnosis.txt")
    with open(diag_path, "r") as file:
        lines = file.readlines()
    split_lines = (line.split() for line in lines)
    diag_map = {parts[0]: parts[1] for parts in split_lines if len(parts) == 2}
    return diag_map


def get_audio_features(source: sf.SoundFile):
    sound, sr = librosa.load(source)


diags = get_diagnoisis_map()
