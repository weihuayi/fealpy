
from typing import overload, Optional, Tuple

from ..backend import backend_manager as bm
from ..backend import TensorLike as _DT
from ..backend import Size
from .sparse_tensor import SparseTensor
from .coo_tensor import COOTensor
from .csr_tensor import CSRTensor

from .ops import spdiags, speye



@overload
def coo_matrix(arg1: _DT, /, itype=None) -> COOTensor: ...
@overload
def coo_matrix(arg1: SparseTensor, /) -> COOTensor: ...
@overload
def coo_matrix(arg1: Size, /, *, itype=None, ftype=None, device=None) -> COOTensor: ...
@overload
def coo_matrix(arg1: Tuple[_DT, Tuple[_DT, ...]], /, *,
               shape: Optional[Size] = None) -> COOTensor: ...
def coo_matrix(arg1, /, *,
               shape: Optional[Size] = None,
               itype=None, ftype=None, device=None) -> COOTensor:
    """A sparse matrix in COOrdinate format.
    (A Scipy-like API to generate sparse tensors without batch.)

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
        itype (dtype | None, optional): Scalar type of indices
        ftype (dtype | None, optional): Scalar type of data
        device (str | device | None, optional): _description_
    """
    if isinstance(arg1, _DT):
        indices_tuple = bm.nonzero(arg1)
        indices = bm.stack(indices_tuple, axis=0)
        if itype is not None:
            indices = bm.astype(indices, itype)
        values = bm.copy(arg1[indices_tuple])
        return COOTensor(indices, values, arg1.shape)

    elif isinstance(arg1, (COOTensor, CSRTensor)):
        return arg1.tocoo()

    elif isinstance(arg1, (tuple, list)):
        if isinstance(arg1[0], int):
            ndim = len(arg1)
            indices = bm.empty((ndim, 0), dtype=itype, device=device)
            values = bm.empty((0,), dtype=ftype, device=device)
            return COOTensor(indices, values, spshape=arg1)

        elif isinstance(arg1[0], _DT) or arg1[0] is None:
            assert len(arg1) == 2
            values = arg1[0] # non-zero elements
            indices = bm.stack(arg1[1], axis=0)
            return COOTensor(indices, values, shape)

    raise TypeError(f"Error: Illegal combination of parameters")


@overload
def csr_matrix(arg1: _DT, /, itype=None) -> CSRTensor: ...
@overload
def csr_matrix(arg1: SparseTensor, /) -> CSRTensor: ...
@overload
def csr_matrix(arg1: Size, /, *, itype=None, ftype=None, device=None) -> CSRTensor: ...
@overload
def csr_matrix(arg1: Tuple[_DT, Tuple[_DT, _DT]], /, *,
               shape: Optional[Size] = None) -> CSRTensor: ...
@overload
def csr_matrix(arg1: Tuple[_DT, _DT, _DT], /, *,
               shape: Optional[Size] = None) -> CSRTensor: ...
def csr_matrix(arg1,
               shape: Optional[Size] = None,
               itype=None, ftype=None, device=None) -> CSRTensor:
    """Compressed Sparse Row matrix.
    (A Scipy-like API to generate sparse tensors without batch.)

    This can be instantiated in several ways:
        csr_matrix(D)
            where D is a 2-D tensor

        csr_matrix(S)
            with another sparse tensor S (equivalent to S.tocsr())

        csr_matrix((M, N), [dtype])
            to construct an empty tensor with shape (M, N).

        csr_matrix((data, (row, col)), [shape=(M, N)])
            where data, row and col satisfy the relationship a[row[k], col[k]] = data[k].

        csr_matrix((data, indices, indptr), [shape=(M, N)])
            is the standard CSR representation where the column indices\
            for row i are stored in indices[indptr[i]:indptr[i+1]]\
            and their corresponding values are stored in data[indptr[i]:indptr[i+1]]

    Parameters:
        arg1 (_type_): _description_
        shape (Size): _description_
        itype (dtype | None, optional): Scalar type of indices
        ftype (dtype | None, optional): Scalar type of data
        device (str | device | None, optional): _description_
    """
    if itype is None:
        itype = bm.int64

    if isinstance(arg1, _DT): # From a dense tensor
        indices_tuple = bm.nonzero(arg1)
        indices = bm.stack(indices_tuple, axis=0)
        if itype is not None:
            indices = bm.astype(indices, itype)
        values = bm.copy(arg1[indices_tuple])
        return COOTensor(indices, values, arg1.shape).tocsr()

    elif isinstance(arg1, SparseTensor): # From another sparse tensor
        return arg1.tocsr()

    elif isinstance(arg1, (tuple, list)):
        if isinstance(arg1[0], int): # Build an empty sparse tensor
            assert len(arg1) == 2
            indptr = bm.zeros((arg1[0]+1,), dtype=itype, device=device)
            indices = bm.empty((0,), dtype=itype, device=device)
            data = bm.empty((0,), dtype=ftype, device=device)
            return CSRTensor(indptr, indices, data, spshape=arg1)

        elif isinstance(arg1[0], _DT) or arg1[0] is None:
            if len(arg1) == 2: # From a COO-like format
                values = arg1[0]
                indices = bm.stack(arg1[1], axis=0)
                return COOTensor(indices, values, shape).tocsr()
            elif len(arg1) == 3: # From CSR data directly
                data, indices, indptr = tuple(arg1)
                return CSRTensor(indptr, indices, data, shape)

    raise ValueError(f"Error: Illegal combination of parameters")


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
