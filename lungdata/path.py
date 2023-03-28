import os

SAMPLES_PATH = os.environ["LUNG_SOUND_PATH"]

META_PATH = os.environ.get("LUNG_META_PATH")
if META_PATH is None:
    module_dir = os.path.dirname(__file__)
    META_PATH = os.path.join(module_dir, "meta")
