import os

sound_env = "LUNG_SOUND_PATH"
try:
    SAMPLES_PATH = os.environ[sound_env]
except KeyError:
    print(f"Warning: {sound_env} environmet varible is not set")
    default_dir = "db"
    print(f"using default location for .wav files: ./{default_dir}")
    SAMPLES_PATH = os.path.join(os.getcwd(), "db")
    print(f"absolute path: {SAMPLES_PATH}")

META_PATH = os.environ.get("LUNG_META_PATH")
if META_PATH is None:
    module_dir = os.path.dirname(__file__)
    META_PATH = os.path.join(module_dir, "meta")
