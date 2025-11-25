from pathlib import Path


def mbed_dirpath(directory: Path) -> Path:
    return directory / ".mbed"


def make_mbed_dir(directory: Path) -> Path:
    mbed_path = mbed_dirpath(directory)
    mbed_path.mkdir(exist_ok=True)
    return mbed_path
