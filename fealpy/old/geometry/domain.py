
from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np

from .sizing_function import huniform


@dataclass
class Domain:
    hmin: float = 0.1
    hmax: float = field(default=None)
    GD: int = field(default=2)
    fh: Callable[..., np.ndarray] = huniform
