from .builder import build_ext
from pathlib import Path

def create_directory(path):
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)  # automatically create the directory if it does not exist 

