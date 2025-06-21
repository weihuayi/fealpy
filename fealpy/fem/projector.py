
from typing import Union, Callable, Optional, Any, TypeVar, Tuple, Dict

from ..typing import TensorLike, CoefLike
from ..functionspace.space import FunctionSpace as _FS

class Projector():
    """
    """
    def __init__(self, space0: _FS, space1: _FS):
        self.space0 = space0
        self.space1 = space1

    def __call__(self, uh):
        raise NotImplementedError

