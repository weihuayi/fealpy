from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
import taichi as ti

EntityName = Literal['cell', 'face', 'edge', 'node']
Entity = TypeVar('Entity') 
Field = TypeVar('Field')
Index = Union[Field, int, slice]

_int_func = Callable[..., int]
_S = slice(None, None, None)

def estr2dim(ds, estr: str) -> int:
    """
    """
    if estr == 'cell':
        return ds.top_dimension()
    elif estr == 'face':
        return ds.top_dimension() - 1
    elif estr == 'edge':
        return 1
    elif estr == 'node':
        return 0
    else:
        raise KeyError(f'{estr} is not a valid entity attribute.')
