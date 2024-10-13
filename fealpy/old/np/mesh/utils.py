from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from .. import logger


EntityName = Literal['cell', 'face', 'edge', 'node']
Index = Union[NDArray, int, slice]
_int_func = Callable[..., int]
_dtype = np.dtype

_S = slice(None, None, None)
_T = TypeVar('_T')

def mesh_top_csr(entity: NDArray, num_targets: int, location: Optional[NDArray]=None, *,
                 dtype: Optional[_dtype]=None) -> NDArray:
    r"""CSR format of a mesh topology relaionship matrix."""

    if entity.ndim == 1: # for polygon case
        if location is None:
            raise ValueError('location is required for 1D entity (usually for polygon mesh).')
        crow = location
    elif entity.ndim == 2: # for homogeneous case
        crow = np.arange(
            entity.size(0) + 1, dtype=entity.dtype
        ).mul_(entity.size(1))
    else:
        raise ValueError('dimension of entity must be 1 or 2.')

    return csr_matrix(
        crow,
        entity.reshape(-1),
        np.ones(entity.numel(), dtype=dtype),
        size=(entity.size(0), num_targets),
        dtype=dtype
    )

def estr2dim(ds, estr: str) -> int:
    """
    Convert an entity string to its corresponding dimension.

    Parameters:
        ds: The mesh data structure which has a method `top_dimension()`.
        estr (str): Entity string representing the entity type. 
                    Should be one of 'cell', 'face', 'edge', or 'node'.

    Returns:
        int: The dimension of the entity.
             - 'cell' returns the top dimension of the mesh data structure.
             - 'face' returns the top dimension minus one.
             - 'edge' returns 1.
             - 'node' returns 0.

    Raises:
        KeyError: If the entity string `estr` is not one of the recognized entity attributes.
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
    
def edim2entity(dict_: Dict, edim: int, index=None):
    r"""Get entity tensor by its top dimension. Returns None if not found."""
    if edim in dict_:
        et = dict_[edim]
        if index is None:
            return et
        else: # TODO: finish this
            if et.ndim == 1:
                raise RuntimeError("index is not supported for flattened entity.")
            return et[index]
    else:
        logger.info(f'entity {edim} is not found and a NoneType is returned.')
        return None
    
# import numpy as np

# def edim2entity(dict_: dict, edim: int, index=None):
#     """
#     Get entity array by its top dimension. Returns None if not found.
    
#     Parameters:
#     - dict_: Dictionary mapping dimensions to entity arrays.
#     - edim: Integer representing the topological dimension of the entity.
#     - index: Optional index or indices to extract from the entity array. If not provided, returns the full array.
    
#     Returns:
#     - Entity array or a subset of it based on the provided index, or None if the entity is not found.
#     """
#     if edim in dict_:
#         et = dict_[edim]
#         if index is None:
#             return et
#         else:
#             if et.ndim == 1:
#                 raise RuntimeError("index is not supported for flattened entity.")
#             # Ensure index is a tuple for multi-dimensional indexing, even if it's a single index
#             if not isinstance(index, tuple):
#                 index = (index,)
#             return et[tuple(index)]
#     else:
#         print(f'Entity {edim} is not found and None is returned.')  # Replacing logger with print for simplicity
#         return None
    
def edim2node(mesh, etype_dim: int, index=None, dtype=None) -> NDArray:
    r"""Get the <entiry>_to_node sparse matrix by entity's top dimension."""
    entity = edim2entity(mesh.storage(), etype_dim, index)
    location = getattr(entity, 'location', None)
    NN = mesh.count('node')
    if NN <= 0:
        raise RuntimeError('No valid node is found in the mesh.')
    return mesh_top_csr(entity, NN, location, dtype=dtype)

def ranges(nv, start = 0):
    shifts = np.cumsum(nv)
    id_arr = np.ones(shifts[-1], dtype=np.int_)
    id_arr[shifts[:-1]] = -np.asarray(nv[:-1])+1
    id_arr[0] = start
    return id_arr.cumsum()

def enable_csr(fn: Callable[..., NDArray]):
    """
    Decorator to make a function generating neighborhood information matrix
    support CSR (Compressed Sparse Row) matrix output.

    Parameters:
        fn (Callable[..., NDArray]): The function to be decorated.

    Returns:
        wrapped (Callable[..., Union[NDArray, csr_matrix]]): The wrapped function
        which can return either a dense ndarray or a sparse CSR matrix based
        on the `return_sparse` parameter.
    """
    @overload
    def wrapped(self, *args, return_sparse: Literal[False], **kwargs) -> NDArray: ...

    @overload
    def wrapped(self, *args, return_sparse: Literal[True], **kwargs) -> csr_matrix: ...

    def wrapped(self, *args, return_sparse=False, **kwargs):
        ret = fn(self, *args, **kwargs)
        if return_sparse:
            return arr_to_csr(arr=ret, reversed=False)
        else:
            return ret
    return wrapped


def arr_to_csr(arr: NDArray, n_col: Optional[int]=None,
               reversed=False, return_local=False, dtype=np.bool_):
    """
    Convert neighbor information matrix to sparse type.

    Parameters:
        arr (NDArray): The input ndarray representing the neighbor information matrix.
        n_col (int, optional): Number of columns of the output sparse matrix.
           For example, when converting `cell_to_edge` to sparse,
           `n_col` should be the number of edges, which will be
           the number of columns of the output sparse matrix.
           If not provided, `max(arr) + 1` is used.
        reversed (bool, optional): If True, transpose the sparse matrix, reversing the
                                   relationship. Default is False.
        return_local (bool, optional): Additional parameter to control the output.
                                       Default is False.
        dtype (np.dtype, optional): The data type of the output values. Default
        is np.bool_.

    Returns:
        csr_matrix: The converted sparse CSR matrix.

    Raises:
        ValueError: If the input array does not have 2 dimensions.
    """
    if arr.ndim != 2:
        raise ValueError("Can only tackle array with 2 dimensions.")
    nr, nv = arr.shape

    if n_col is not None:
        nc = n_col
    else:
        nc = np.max(arr) + 1

    if not return_local:
        val = np.ones(nr*nv, dtype=np.bool_)
    else:
        val = ranges(nv*np.ones(nr, dtype=dtype), start=1)

    if not reversed:
        sp = csr_matrix(
            (
                val,
                (
                    np.repeat(range(nr), nv),
                    arr.flat
                )
            ),
            shape=(nr, nc)
        )
        return sp
    else:
        sp = csr_matrix(
            (
                val,
                (
                    arr.flat,
                    np.repeat(range(nr), nv)
                )
            ),
            shape=(nc, nr)
        )
        return sp

