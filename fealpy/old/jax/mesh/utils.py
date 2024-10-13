from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import jax
import jax.numpy as jnp 
from jax.experimental.sparse import BCOO
from jax import config

config.update("jax_enable_x64", True)

from .. import logger

Array = jax.Array
Index = Union[Array, int, slice]
EntityName = Literal['cell', 'face', 'edge', 'node']
_int_func = Callable[..., int]
_dtype = jnp.dtype
_device = jax.Device

_S = slice(None, None, None)
_T = TypeVar('_T')
_default = object()

def mesh_top_csr(entity: Array, num_targets: int, location: Optional[Array]=None, *,
                 dtype: Optional[_dtype]=None) -> Array:
    r"""CSR format of a mesh topology relaionship matrix."""
    device = entity.device

    if entity.ndim == 1: # for polygon case
        if location is None:
            raise ValueError('location is required for 1D entity (usually for polygon mesh).')
        crow = location
    elif entity.ndim == 2: # for homogeneous case
        crow = jnp.arange(
            entity.shape[0] + 1, dtype=entity.dtype)*(entity.shape[1])
    else:
        raise ValueError('dimension of entity must be 1 or 2.')
    data = jnp.ones(jnp.prod(entity.shape), dtype=dtype)
    return BCOO((data, (crow, entity.reshape(-1))),
                     shape=(entity.size(0), num_targets),
                     ).todense()

def estr2dim(mesh, estr: str) -> int:
    if estr == 'cell':
        return mesh.top_dimension()
    elif estr == 'face':
        TD = mesh.top_dimension()
        return TD - 1
    elif estr == 'edge':
        return 1
    elif estr == 'node':
        return 0
    else:
        raise KeyError(f'{estr} is not a valid entity name in FEALPy.')


def edim2entity(dict_: Dict, edim: int, index=None):
    r"""Get entity Array by its top dimension. Returns None if not found."""
    if edim in dict_:
        et = dict_[edim]
        if index is None:
            return et
        else: # TODO: finish this for homogeneous mesh
            return et[index]
    else:
        logger.info(f'entity {edim} is not found and a NoneType is returned.')
        return None


def edim2node(mesh, etype_dim: int, index=None, dtype=None) -> Array:
    r"""Get the <entiry>_to_node sparse matrix by entity's top dimension."""
    entity = edim2entity(mesh.storage(), etype_dim, index)
    location = getattr(entity, 'location', None)
    NN = mesh.count('node')
    if NN <= 0:
        raise RuntimeError('No valid node is found in the mesh.')
    return mesh_top_csr(entity, NN, location, dtype=dtype)
