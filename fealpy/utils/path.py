
from pathlib import Path

def project_root() -> Path:
    """
    Return the absolute path of the FEALPy project root directory.

    The function assumes the following directory structure:

        fealpy/
        ├── fealpy/        # source code
        ├── data/          # data files
        └── example/       # examples

    Returns:
        Path: the absolute path to the project root (directory containing 'fealpy' and 'data').
    """
    return Path(__file__).resolve().parent.parent.parent


def data_root() -> Path:
    """
    Return the absolute path of the ``fealpy/data`` directory.

    Returns:
        Path: to the data directory.
    """
    return project_root() / "data"


def get_data_path(*subpaths: str) -> Path:
    """
    Get the absolute path to a data file inside the ``fealpy/data`` directory.

    Parameters:
        *subpaths (str): One or more path components under ``fealpy/data``.

    Examples:
        >>> mesh_path = get_data_path("mesh", "tri.inp")
        >>> print(mesh_path)
        /absolute/path/to/fealpy/data/mesh/tri.inp

    Returns
        Path: Absolute path to the requested file or directory.

    Raises:
        FileNotFoundError: If the constructed path does not exist.
    """
    path = data_root().joinpath(*subpaths).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    return path
