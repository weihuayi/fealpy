from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
import jax
import jax.numpy as jnp 

Array = jax.Array
Index = Union[Array, int, slice]
EntityName = Literal['cell', 'face', 'edge', 'node']
_int_func = Callable[..., int]
_dtype = jnp.dtype
_device = jax.Device

_S = slice(None, None, None)
_T = TypeVar('_T')
_default = object()

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
