from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
import taichi as ti

from ti.types import template as Template 
from ti.types import ndarray as NDArray

EntityName = Literal['cell', 'cell_location', 'face', 'face_location', 'edge']
_int_func = Callable[..., int]
_T = TypeVar('_T') # 用创建通用类型变量
_default = object()

def mesh_top_csr(entity: Template, num_targets: int, location: Optional[Template]=None, *,
                 dtype=None) -> Template:
    r"""CSR format of a mesh topology relaionship matrix."""
    pass

def entity_str2dim(ds, etype: str) -> int:
    if etype == 'cell':
        return ds.top_dimension()
    elif etype == 'cell_location':
        return -ds.top_dimension()
    elif etype == 'face':
        TD = ds.top_dimension()
        return TD - 1
    elif etype == 'face_location':
        TD = ds.top_dimension()
        return -TD + 1
    elif etype == 'edge':
        return 1
    elif etype == 'node':
        return 0
    else:
        raise KeyError(f'{etype} is not a valid entity attribute.')


def entity_dim2field(ds, etype_dim: int, index=None, *, default=_default):
    r"""Get entity tensor by its top dimension."""
    if etype_dim in ds._entity_storage:
        et = ds._entity_storage[etype_dim]
        if index is None:
            return et
        else:
            if et.ndim == 1:
                raise RuntimeError("index is not supported for flattened entity.")
            return et[index]
    else:
        if default is not _default:
            return default
        raise ValueError(f'{etype_dim} is not a valid entity attribute index '
                         f"in {ds.__class__.__name__}.")


def entity_dim2node(ds, etype_dim: int, index=None, dtype=None) -> Tensor:
    r"""Get the <entiry>_to_node sparse matrix by entity's top dimension."""
    pass
