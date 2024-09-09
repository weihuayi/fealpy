
from typing import overload, Optional, Tuple

from ..backend import TensorLike as _DT
from ..backend import Size
from .sparse_tensor import SparseTensor
from .coo_tensor import COOTensor
from .csr_tensor import CSRTensor


@overload
def coo_matrix(arg1: _DT, *,
               dims: Optional[int]=None,
               dtype=None,
               copy=False) -> COOTensor: ...
@overload
def coo_matrix(arg1: SparseTensor, *,
               dtype=None,
               copy=False) -> COOTensor: ...
@overload
def coo_matrix(arg1: Size, *,
               dims: Optional[int]=None,
               dtype=None,
               copy=False) -> COOTensor: ...
@overload
def coo_matrix(arg1: Tuple[_DT, Tuple[_DT, ...]], *,
               shape: Optional[Size]=None,
               dtype=None,
               copy=False) -> COOTensor: ...
def coo_matrix(arg1, *,
               shape: Optional[Size]=None,
               dims: Optional[int]=None,
               dtype=None,
               copy=False) -> COOTensor:
    """A sparse matrix in COOrdinate format.

    Also known as the 'ijv' or 'triplet' format.

    This can be instantiated in several ways:
        coo_matrix(D)
            where D is a 2-D array

        coo_matrix(S)
            with another sparse array or matrix S (equivalent to S.tocoo())

        coo_matrix((M, ...), [dtype])
            to construct an empty matrix with shape (M, ...)
            dtype is optional, defaulting to dtype='d'.

        coo_matrix((data, (i, ...)), [shape=(M, ...)])
            to construct from at least two arrays:
                1. data[:]   the entries of the matrix, in any order
                2. i[:]      the row indices of the matrix entries
                3. j[:]      the column indices of the matrix entries
                4. more indices can be passed as well for higher sparse dimensions

            Where ``A[i[k], j[k]] = data[k]`` (2D case).  When shape is not
            specified, it is inferred from the index arrays

    Parameters:
        arg1 (_type_): _description_
        shape (Size | None, optional): _description_
        dims (int | None, optional): _description_
        dtype (dtype | None, optional): _description_
        device (str | device | None, optional): _description_
        copy (bool, optional): _description_. Defaults to False.

    Raises:
        TypeError: _description_

    Returns:
        COOTensor: _description_
    """
    if isinstance(arg1, _DT):
        pass
    elif isinstance(arg1, (COOTensor, CSRTensor)):
        pass
    elif isinstance(arg1, (tuple, list)):
        if isinstance(arg1[0], _DT):
            pass
        elif isinstance(arg1[0], int):
            pass
        else:
            raise TypeError(f"Unsupported type {type(arg1[0])}")
    else:
        raise TypeError(f"Unsupported type {type(arg1)}")


def csr_matrix(arg1, shape: Size, *,
               dtype=None,
               copy=False) -> CSRTensor:
    """_summary_

    Parameters:
        arg1 (_type_): _description_
        shape (Size): _description_
        dtype (dtype | None, optional): _description_
        device (str | device | None, optional): _description_
        copy (bool, optional): _description_. Defaults to False.

    Raises:
        TypeError: _description_

    Returns:
        CSRTensor: _description_
    """
    raise NotImplementedError



# NOTE: APIs for Sparse Tensors

# 1. Data Fetching:
# itype, ftype, nnz,
# shape, dense_shape, sparse_shape,
# ndim, dense_ndim, sparse_ndim,
# size,
# values_context,
# (COO) indices, values,
# (CSR) crow, col, values

# 2. Data Type & Device Management:
# astype, device_put

# 3. Format Conversion:
# to_dense (=toarray), tocsr, tocoo

# 4. Object Conversion:
# to_scipy, from_scipy,

# 5. Manipulation:
# copy, coalesce, reshape, flatten, ravel,
# tril, triu,
# concat

# 6. Arithmetic Operations:
# add, sub, mul, div, pow, neg, matmul
