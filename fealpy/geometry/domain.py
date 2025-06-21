
from dataclasses import dataclass, field
from typing import Callable, Any
from ..backend import backend_manager as bm
from .sizing_function import huniform


@dataclass
class Domain:
    hmin: float = 0.1
    hmax: float = field(default=None)
    GD: int = field(default=2)
    fh: Callable[..., Any] = huniform
