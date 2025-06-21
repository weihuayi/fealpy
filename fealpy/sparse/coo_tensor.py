
from typing import Optional, Union, overload, Tuple, Sequence
from math import prod

from ..backend import TensorLike, Number, Size
from ..backend import backend_manager as bm
from .sparse_tensor import SparseTensor
from .utils import (
    flatten_indices, check_shape_match, check_spshape_match
)
from ._spspmm import spspmm_coo
from ._spmm import spmm_coo


class COOTensor(SparseTensor):
    def __init__(self, indices: TensorLike, values: Optional[TensorLike],
                 spshape: Optional[Size] = None, *,
                 is_coalesced: Optional[bool] = None):
        """
        Initialize COO format sparse tensor.

        Parameters:
            indices (Tensor): indices of non-zero elements, shaped (D, nnz).
                Where D is the number of sparse dimension, and nnz is the number
                of non-zeros.
            values (Tensor | None): non-zero elements, shaped (..., nnz).
            spshape (Size | None, optional): shape in the sparse dimensions.
        """
        self._indices = indices
        self._values = values
        self.is_coalesced = is_coalesced
        self._check(indices, values)

        if spshape is None:
            self._spshape = tuple(bm.tolist(bm.max(indices, axis=1) + 1))
        else:
            # total ndim should be equal to sparse_ndim + dense_dim
            if len(spshape) != indices.shape[0]:
                raise ValueError(
                    f"length of sparse shape ({len(spshape)}) "
                    f"must match the size of indices in dim-0 ({indices.shape[0]})"
                )
            self._spshape = tuple(spshape)

    def _check(self, indices: TensorLike, values: Optional[TensorLike]):
        if not isinstance(indices, TensorLike):
            raise TypeError(f"indices must be a Tensor, but got {type(indices)}")
        if indices.ndim != 2:
            raise ValueError(f"indices must be a 2D tensor, but got {indices.ndim}D")

        if isinstance(values, TensorLike):
            if values.ndim < 1:
                raise ValueError(f"values must be at least 1D, but got {values.ndim}D")

            # The last dim of values must match the last dim of indices.
            if values.shape[-1] != indices.shape[1]:
                raise ValueError(f"values must have the same size as indices ({indices.shape[1]}) "
                                 "in the last dimension (number of non-zero elements), "
                                 f"but got {values.shape[-1]}")
        elif values is None:
            pass
        else:
            raise TypeError(f"values must be a Tensor or None, but got {type(values)}")

    def __repr__(self) -> str:
        return f"COOTensor(indices={self._indices}, values={self._values}, shape={self.shape})"

    ### 1. Data Fetching ###
    @property
    def itype(self): return self._indices.dtype

    @property
    def nnz(self): return self._indices.shape[1]

    @property
    def indices(self) -> TensorLike:
        """Return the indices of the non-zero elements."""
        return self._indices

    @property
    def values(self) -> Optional[TensorLike]:
        """Return the non-zero elements."""
        return self._values

    @property
    def row(self): return self._indices[0] # scipy convention

    @property
    def col(self): return self._indices[1] # scipy convenstion

    @property
    def data(self): return self._values # scipy convention

    @property
    def nonzero_slice(self) -> Tuple[Union[slice, TensorLike]]:
        slicing = [self._indices[i] for i in range(self.sparse_ndim)]
        return (slice(None),) * self.dense_ndim + tuple(slicing)

    ### 2. Data Type & Device Management ###
    def astype(self, dtype=None, /, *, copy=True):
        if self._values is None:
            values = bm.ones(self.nnz, dtype=dtype)
        else:
            values = bm.astype(self._values, dtype, copy=copy)

        return COOTensor(self._indices, values, self._spshape,
                         is_coalesced=self.is_coalesced)

    def device_put(self, device=None, /):
        return COOTensor(bm.device_put(self._indices, device),
                         bm.device_put(self._values, device),
                         self._spshape,
                         is_coalesced=self.is_coalesced)

    ### 3. Format Conversion ###
    def to_dense(self, *, fill_value: Union[Number, bool] = 1, dtype=None) -> TensorLike:
        if self.values is None:
            dtype = bm.float64 if (dtype is None) else dtype
            context = {"dtype": dtype, "device": bm.get_device(self.indices)}
            src = bm.full((1,) * (self.dense_ndim + 1), fill_value, **context)
            src = bm.broadcast_to(src, self.dense_shape + (self.nnz,))
        else:
            src = self.values if (dtype is None) else bm.astype(self.values, dtype)
            context = {"dtype": src.dtype, "device": bm.get_device(src)}

        dense_tensor = bm.zeros(self.dense_shape + (prod(self._spshape),), **context)
        flattened = flatten_indices(self._indices, self._spshape)[0]
        dense_tensor = bm.index_add(dense_tensor, flattened, src, axis=-1)

        return dense_tensor.reshape(self.shape)

    def tocoo(self, *, copy=False):
        if copy:
            return self.copy()
        return self

    def tocsr(self, *, copy=False):
        from .csr_tensor import CSRTensor
        # try:
        #     crow, col, values = bm.coo_tocsr(self.indices, self.values, self.sparse_shape)
        #     return CSRTensor(crow, col, values, spshape=self._spshape)
        # except (AttributeError, NotImplementedError):
        #     pass

        count = bm.bincount(self._indices[0], minlength=self._spshape[0])
        crow = bm.cumsum(count, axis=0)
        crow = bm.concat([bm.tensor([0], **bm.context(crow)), crow])
        order = bm.argsort(self._indices[0], stable=True)
        new_col = bm.copy(self._indices[-1, order])

        if self.values is None:
            new_values = None
        else:
            new_values = self.values[..., order]
            new_values = bm.copy(new_values) if copy else new_values

        return CSRTensor(crow, new_col, new_values, spshape=self._spshape)

    ### 4. Object Conversion ###
    def to_scipy(self):
        from scipy.sparse import coo_matrix
        if self.dense_ndim != 0:
            raise ValueError("Only COOTensor with 0 dense dimension "
                             "can be converted to scipy sparse matrix")


        return coo_matrix(
            (bm.to_numpy(self.values), bm.to_numpy(self.indices)),
            shape = self.sparse_shape
        )

    @classmethod
    def from_scipy(cls, mat, /):
        indices = bm.stack([bm.from_numpy(mat.row), bm.from_numpy(mat.col)], axis=0)
        values = bm.from_numpy(mat.data)
        return cls(indices, values, mat.shape)

    ### 5. Manipulation ###
    def copy(self):
        if self._values is None:
            return COOTensor(bm.copy(self._indices), None, self._spshape)
        return COOTensor(bm.copy(self._indices), bm.copy(self._values), self._spshape)

    def coalesce(self, accumulate: bool=True) -> 'COOTensor':
        if self.is_coalesced or self.nnz == 0:
            return self

        order = bm.lexsort(tuple(reversed(self._indices)))
        sorted_indices = self._indices[:, order]
        unique_mask = bm.concat([
            bm.ones((1, ), dtype=bool, device=bm.get_device(sorted_indices)),
            bm.any(sorted_indices[:, 1:] - sorted_indices[:, :-1], axis=0)
        ], axis=0)
        new_indices = bm.copy(sorted_indices[..., unique_mask])

        if self._values is not None:
            add_index = bm.cumsum(unique_mask, axis=0) - 1
            sorted_values = self._values[..., order]
            new_values = bm.zeros_like(sorted_values[..., unique_mask])
            new_values = bm.index_add(new_values, add_index, sorted_values, axis=-1)

        else:
            if accumulate:
                unique_location = bm.concat([
                    bm.nonzero(unique_mask)[0],
                    bm.tensor([len(unique_mask)], **bm.context(self._indices))
                ], axis=0)
                new_values = unique_location[1:] - unique_location[:-1]

            else:
                new_values = None

        return COOTensor(new_indices, new_values, self.sparse_shape, is_coalesced=True)

    @overload
    def reshape(self, shape: Size, /) -> 'COOTensor': ...
    @overload
    def reshape(self, *shape: int) -> 'COOTensor': ...
    def reshape(self, *shape) -> 'COOTensor':
        pass

    def ravel(self):
        spshape = self.sparse_shape
        new_indices = flatten_indices(self._indices, spshape)
        return COOTensor(new_indices, self._values, (prod(spshape),))

    def flatten(self):
        spshape = self.sparse_shape
        new_indices = flatten_indices(self._indices, spshape)
        if self._values is None:
            values = None
        else:
            values = bm.copy(self._values)
        return COOTensor(new_indices, values, (prod(spshape),))

    @property
    def T(self):
        _indices = self._indices
        _spshape = self._spshape

        if self.sparse_ndim == 2:
            new_indices = bm.stack([_indices[1], _indices[0]], axis=0)
            shape = tuple(reversed(_spshape))
        elif self.sparse_ndim >= 3:
            new_indices = bm.concat([_indices[:-2], _indices[-1:], _indices[-2:-1]], axis=0)
            shape = _spshape[:-2] + (_spshape[-1], _spshape[-2])
        else:
            raise ValueError("sparse ndim must be 2 or greater to be transposed, "
                             f"but got {self.sparse_ndim}")
        return COOTensor(new_indices, self._values, shape)

    def partial(self, index: Union[TensorLike, slice], /):
        new_indices = bm.copy(self.indices[:, index])
        new_values = self.values

        if new_values is not None:
            new_values = bm.copy(new_values[..., index])

        return COOTensor(new_indices, new_values, self._spshape)

    def tril(self, k: int = 0) -> 'COOTensor':
        indices = self.indices
        tril_loc = (indices[-2] + k) >= indices[-1]
        return self.partial(tril_loc)

    def triu(self, k: int = 0) -> 'COOTensor':
        indices = self.indices
        triu_loc = (indices[-1] - k) >= indices[-2]
        return self.partial(triu_loc)

    @classmethod
    def concat(cls, coo_tensors: Sequence['COOTensor'], /, *, axis: int=0) -> 'COOTensor':
        if len(coo_tensors) == 0:
            raise ValueError("coo_tensors cannot be empty")

        if len(coo_tensors) == 1:
            return coo_tensors[0]

        indices_list = []
        values_list = []
        prev_len = 0

        for coo in coo_tensors:
            indices = bm.copy(coo.indices)
            indices = bm.index_add(
                indices, bm.array([axis], device=indices.device), prev_len,
                axis=0
            )
            indices_list.append(indices)
            values_list.append(bm.copy(coo.values))
            prev_len += coo.sparse_shape[axis]

        new_indices = bm.concat(indices_list, axis=1)
        del indices_list
        new_values = bm.concat(values_list, axis=-1)
        del values_list
        spshape = list(coo_tensors[-1].sparse_shape)
        spshape[axis] = prev_len
        return cls(new_indices, new_values, spshape)

    ### 6. Arithmetic Operations ###
    def neg(self) -> 'COOTensor':
        """Negation of the COO tensor. Returns self if values is None."""
        if self._values is None:
            return self
        else:
            return COOTensor(self._indices, -self._values, self.sparse_shape)

    @overload
    def add(self, other: Union[Number, 'COOTensor'], alpha: Number=1) -> 'COOTensor': ...
    @overload
    def add(self, other: TensorLike, alpha: Number=1) -> TensorLike: ...
    def add(self, other: Union[Number, 'COOTensor', TensorLike], alpha: Number=1) -> Union['COOTensor', TensorLike]:
        """Adds another tensor or scalar to this COOTensor, with an optional scaling factor.

        Parameters:
            other (Number | COOTensor | Tensor): The tensor or scalar to be added.\n
            alpha (int | float, optional): The scaling factor for the other tensor. Defaults to 1.

        Raises:
            TypeError: If the type of `other` is not supported for addition.\n
            ValueError: If the shapes of `self` and `other` are not compatible.\n
            ValueError: If one has value and another does not.

        Returns:
            out (COOTensor | Tensor): A new COOTensor if `other` is a COOTensor,\
            or a Tensor if `other` is a dense tensor.
        """
        if isinstance(other, COOTensor):
            check_shape_match(self.shape, other.shape)
            check_spshape_match(self.sparse_shape, other.sparse_shape)
            new_indices = bm.concat((self._indices, other._indices), axis=1)
            if self._values is None:
                if other._values is None:
                    new_values = None
                else:
                    raise ValueError("self has no value while other does")
            else:
                if other._values is None:
                    raise ValueError("self has value while other does not")
                new_values = bm.concat((self._values, other._values*alpha), axis=-1)
            return COOTensor(new_indices, new_values, self.sparse_shape)

        elif isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            output = other * alpha
            context = bm.context(output)
            output = output.reshape(self.dense_shape + (prod(self._spshape),))
            flattened = flatten_indices(self._indices, self._spshape)[0]

            if self._values is None:
                src = bm.ones((1,) * (self.dense_ndim + 1), **context)
                src = bm.broadcast_to(src, self.dense_ndim + (self.nnz,))
            else:
                src = self._values
            output = bm.index_add(output, flattened, src, axis=-1)

            return output.reshape(self.shape)

        elif isinstance(other, (int, float)):
            new_values = self._values + alpha * other
            return COOTensor(bm.copy(self._indices), new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in addition")

    def mul(self, other: Union[Number, 'COOTensor', TensorLike]) -> 'COOTensor': # TODO: finish this
        """Element-wise multiplication.
        The result COO tensor will share the same indices with
        the original if `other` is a number or a dense tensor.
        """
        if isinstance(other, COOTensor):
            raise NotImplementedError

        elif isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            new_values = bm.copy(other[self.nonzero_slice])
            if self._values is not None:
                new_values = bm.multiply(self._values, new_values)
            return COOTensor(self._indices, new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            if self._values is None:
                raise ValueError("Cannot multiply COOTensor without value with scalar")
            new_values = self._values * other
            return COOTensor(self._indices, new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in multiplication")

    def div(self, other: Union[Number, TensorLike]) -> 'COOTensor':
        """Element-wise division.
        The result COO tensor will share the same indices with
        the original if `other` is a number or a dense tensor.
        """
        if self._values is None:
            raise ValueError("Cannot divide COOTensor without value")

        if isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            new_values = bm.copy(other[self.nonzero_slice])
            new_values = bm.divide(self._values, new_values)
            return COOTensor(self._indices, new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            new_values = self._values / other
            return COOTensor(self._indices, new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in division")

    def pow(self, other: Union[TensorLike, Number]) -> 'COOTensor':
        """Element-wise power of COOTensor.
        The result COO tensor will share the same indices with
        the original if `other` is a number or a dense tensor.
        """
        if self._values is None:
            raise ValueError("Cannot power COOTensor without value with tensor")

        if isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            new_values = bm.power(self._values, other[self.nonzero_slice])
            return COOTensor(self._indices, new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            new_values = self._values ** other
            return COOTensor(self._indices, new_values, self.sparse_shape)

        else:
            raise TypeError(f'Unsupported type {type(other).__name__} in power')

    @overload
    def matmul(self, other: 'COOTensor') -> 'COOTensor': ...
    @overload
    def matmul(self, other: TensorLike) -> TensorLike: ...
    def matmul(self, other: Union['COOTensor', TensorLike]):
        """Matrix-multiply this COOTensor with another tensor.

        Parameters:
            other (COOTensor | Tensor): A 1-D tensor for matrix-vector multiply,
                or a 2-D tensor for matrix-matrix multiply.
                Batched matrix-matrix multiply is available for dimensions
                (*B, M, K) and (*B, K, N). *B means any number of batch dimensions.

        Raises:
            TypeError: If the type of `other` is not supported for matmul.

        Returns:
            out (COOTensor | Tensor): A new COOTensor if `other` is a COOTensor,\
            or a Tensor if `other` is a dense tensor.
        """
        if isinstance(other, COOTensor):
            if (self.values is None) or (other.values is None):
                raise ValueError("Matrix multiplication between COOTensor without "
                                 "value is not implemented now")
            if hasattr(bm, 'csr_spspmm'):
                from .csr_tensor import CSRTensor
                mat1 = self.tocsr()
                mat2 = other.tocsr()
                crow, col, values, spshape = bm.csr_spspmm(
                    mat1.crow, mat1.col, mat1.values, mat1.sparse_shape,
                    mat2.crow, mat2.col, mat2.values, mat2.sparse_shape
                )
                return CSRTensor(crow, col, values, spshape)
            else:
                indices, values, spshape = spspmm_coo(
                    self.indices, self.values, self.sparse_shape,
                    other.indices, other.values, other.sparse_shape,
                )
            return COOTensor(indices, values, spshape).coalesce().tocsr()

        elif isinstance(other, TensorLike):
            if self.values is None:
                raise ValueError()
            if hasattr(bm, 'coo_spmm'):
                return bm.coo_spmm(self._indices, self._values, self._spshape, other)
            else:
                return spmm_coo(self.indices, self.values, self.sparse_shape, other)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in matmul")
