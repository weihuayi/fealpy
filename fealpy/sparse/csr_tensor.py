
from typing import Optional, Union, overload, List,Tuple
from math import prod

from ..backend import TensorLike, Number, Size
from ..backend import backend_manager as bm
from .sparse_tensor import SparseTensor
from .utils import (
    flatten_indices,
    check_shape_match, check_spshape_match
)
from ._spspmm import spspmm_csr
from ._spmm import spmm_csr


class CSRTensor(SparseTensor):
    def __init__(self, crow: TensorLike, col: TensorLike, values: Optional[TensorLike],
                 spshape: Optional[Size]=None) -> None:
        """Initializes CSR format sparse tensor.

        Parameters:
            crow (Tensor): compressed row pointers.
            col (Tensor): column indices of non-zero elements, shaped (nnz,).
                Where nnz is the number of non-zeros.
            values (Tensor | None): non-zero elements, shaped (..., nnz).
            spshape (Size | None, optional): shape in the sparse dimensions.
        """
        self._crow = crow
        self._col = col
        self._values = values

        if spshape is None:
            nrow = crow.shape[0] - 1
            ncol = bm.max(col) + 1
            self._spshape = (nrow, ncol)
        else:
            self._spshape = tuple(spshape)

        self._check(crow, col, values, self._spshape)

    def _check(self, crow: TensorLike, col: TensorLike, values: Optional[TensorLike], spshape: Size):
        if crow.ndim != 1:
            raise ValueError(f"crow must be a 1-D tensor, but got {crow.ndim}")
        if col.ndim != 1:
            raise ValueError(f"col must be a 1-D tensor, but got {col.ndim}")
        if len(spshape) != 2:
                raise ValueError(f"spshape must be a 2-tuple for CSR format, but got {spshape}")

        if spshape[0] != crow.shape[0] - 1:
            raise ValueError(f"crow.shape[0] - 1 must be equal to spshape[0], "
                             f"but got {crow.shape[0] - 1} and {spshape[0]}")

        if isinstance(values, TensorLike):
            if values.ndim < 1:
                raise ValueError(f"values must be at least 1-D, but got {values.ndim}")

            if values.shape[-1] != col.shape[-1]:
                raise ValueError(f"values must have the same size as col ({col.shape[-1]}) "
                                 "in the last dimension (number of non-zero elements), "
                                 f"but got {values.shape[-1]}")
        elif values is None:
            pass
        else:
            raise ValueError(f"values must be a Tensor or None, but got {type(values)}")

    def __repr__(self) -> str:
        return f"CSRTensor(crow={self._crow}, col={self._col}, "\
               + f"values={self._values}, shape={self.shape})"

    ### 1. Data Fetching ###
    @property
    def itype(self): return self._crow.dtype

    @property
    def nnz(self): return self._col.shape[-1]

    @property
    def nonzero_slice(self) -> Tuple[Union[slice, TensorLike]]:
        nonzero_row = bm.zeros(len(self._values),dtype=bm.int64)
        nonzero_col = bm.zeros(len(self._values),dtype=bm.int64)

        for i in range(1, self._crow.shape[0]):
                start = self._crow[i - 1]
                end = self._crow[i]
                nonzero_row[start:end] = bm.zeros(end - start) + i-1
                nonzero_col[start:end] = self._col[start:end]

        return nonzero_row, nonzero_col

    def crow(self) -> TensorLike:
        """Return the row location of non-zero elements."""
        return self._crow

    def row(self) -> TensorLike:
        """Generate the row id of non-zero elements."""
        crow = self.crow()
        n_row = crow.shape[0] - 1
        return bm.repeat(
            bm.arange(n_row, dtype=crow.dtype, device=bm.get_device(crow)),
            crow[1:] - crow[:-1]
        )

    def col(self) -> TensorLike:
        """Return the column of non-zero elements."""
        return self._col

    def values(self) -> Optional[TensorLike]:
        """Return the non-zero elements"""
        return self._values

    ### 2. Data Type & Device Management ###
    def astype(self, dtype=None, /, *, copy=True):
        if self._values is None:
            values = bm.ones(self.nnz, dtype=dtype)
        else:
            values = bm.astype(self._values, dtype, copy=copy)

        return CSRTensor(self._crow, self._col, values, self._spshape)

    def device_put(self, device=None, /):
        return CSRTensor(bm.device_put(self._crow, device),
                         bm.device_put(self._col, device),
                         bm.device_put(self._values, device),
                         self._spshape)

    ### 3. Format Conversion ###
    def to_dense(self, *, fill_value: Number=1.0) -> TensorLike:
        if self._values is None:
            context = dict(dtype=bm.float64, device=bm.get_device(self._crow))
        else:
            context = self.values_context()

        index_context = {'dtype': self._crow.dtype, 'device': bm.get_device(self._crow)}

        count = self._crow[1:] - self._crow[:-1]
        nrow = self._crow.shape[0] - 1
        row = bm.repeat(bm.arange(nrow, **index_context), count)
        indices = bm.stack([row, self._col], axis=0)

        dense_tensor = bm.zeros(self.dense_shape + (prod(self._spshape),), **context)
        flattened = flatten_indices(indices, self._spshape)[0]

        if self._values is None:
            src = bm.full((1,) * (self.dense_ndim + 1), fill_value, **context)
            src = bm.broadcast_to(src, self.dense_shape + (self.nnz,))
        else:
            src = self._values
        dense_tensor = bm.index_add(dense_tensor, flattened, src, axis=-1)

        return dense_tensor.reshape(self.shape)

    def tocoo(self, *, copy=False):
        from .coo_tensor import COOTensor
        count = self._crow[1:] - self._crow[:-1]
        nrow = self._crow.shape[0] - 1
        row = bm.repeat(bm.arange(nrow, device=bm.get_device(count)), count)
        indices = bm.stack([row, self._col], axis=0)
        new_values = bm.copy(self._values) if copy else self._values
        return COOTensor(indices, new_values, self.sparse_shape)

    def tocsr(self, *, copy=False):
        if copy:
            return CSRTensor(bm.copy(self._crow), bm.copy(self._col),
                             bm.copy(self._values), self._spshape)
        return self

    ### 4. Object Conversion ###
    def to_scipy(self):
        from scipy.sparse import csr_matrix

        if self.dense_ndim != 0:
            raise ValueError("Only CSRTensor with 0 dense dimension "
                             "can be converted to scipy sparse matrix")

        return csr_matrix(
            (bm.to_numpy(self._values), bm.to_numpy(self._col), bm.to_numpy(self._crow)),
            shape = self._spshape
        )

    @classmethod
    def from_scipy(cls, mat, /):
        crow = bm.from_numpy(mat.indptr)
        col = bm.from_numpy(mat.indices)
        values = bm.from_numpy(mat.data)
        return cls(crow, col, values, mat.shape)

    ### 5. Manipulation ###
    def copy(self):
        if self._values is None:
            return CSRTensor(bm.copy(self._crow), bm.copy(self._col),
                             None, self._spshape)
        return CSRTensor(bm.copy(self._crow), bm.copy(self._col),
                         bm.copy(self._values), self._spshape)

    def coalesce(self, accumulate: bool=True) -> 'CSRTensor':
        raise NotImplementedError

    @overload
    def reshape(self, shape: Size, /) -> 'CSRTensor': ...
    @overload
    def reshape(self, *shape: int) -> 'CSRTensor': ...
    def reshape(self, *shape) -> 'CSRTensor':
        pass

    def ravel(self) -> 'CSRTensor':
        pass

    def flatten(self) -> 'CSRTensor':
        pass

    @property
    def T(self):
        raise NotImplementedError

    def partial(self, index: Union[TensorLike, slice]):
        crow = self.crow()
        ZERO = bm.zeros([1], dtype=crow.dtype, device=bm.get_device(crow))
        new_col = bm.copy(self.col()[..., index])
        is_selected = bm.zeros((self.nnz,), dtype=bm.bool, device=bm.get_device(new_col))
        is_selected = bm.set_at(is_selected, index, True)
        selected_cum = bm.concat([ZERO, bm.cumsum(is_selected, axis=0)], axis=0)
        new_nnz_per_row = selected_cum[crow[1:]] - selected_cum[crow[:-1]]
        new_crow = bm.concat([ZERO, bm.cumsum(new_nnz_per_row, axis=0)], axis=0)

        new_values = self.values()

        if new_values is not None:
            new_values = bm.copy(new_values[..., index])

        return CSRTensor(new_crow, new_col, new_values, self._spshape)

    def tril(self, k: int = 0) -> 'CSRTensor':
        tril_loc = (self.row() + k) >= self.col()
        return self.partial(tril_loc)

    def triu(self, k: int = 0) -> 'CSRTensor':
        tril_loc = (self.col() - k) >= self.row()
        return self.partial(tril_loc)

    ### 6. Arithmetic Operations ###
    def neg(self) -> 'CSRTensor':
        """Negation of the CSR tensor. Returns self if values is None."""
        if self._values is None:
            return self
        else:
            return CSRTensor(self._crow, self._col, -self._values, self._spshape)

    @overload
    def add(self, other: Union[Number, 'CSRTensor'], alpha: Number=1) -> 'CSRTensor': ...
    @overload
    def add(self, other: TensorLike, alpha: Number=1) -> TensorLike: ...
    def add(self, other: Union[Number, 'CSRTensor', TensorLike], alpha: Number=1) -> Union['CSRTensor', TensorLike]:
        """Adds another tensor or scalar to this CSRTensor, with an optional scaling factor.

        Parameters:
            other (Number | CSRTensor | Tensor): The tensor or scalar to be added.\n
            alpha (float, optional): The scaling factor for the other tensor. Defaults to 1.0.

        Raises:
            TypeError: If the type of `other` is not supported for addition.\n
            ValueError: If the shapes of `self` and `other` are not compatible.\n
            ValueError: If one has value and another does not.

        Returns:
            out (CSRTensor | Tensor): A new CSRTensor if `other` is a CSRTensor,\
            or a Tensor if `other` is a dense tensor.
        """
        if isinstance(other, CSRTensor):
            check_shape_match(self.shape, other.shape)
            check_spshape_match(self.sparse_shape, other.sparse_shape)

            if (self._values is None) and (not other._values is None):  
                raise ValueError("self has no value while other does")
            elif (not self._values is None) and (other._values is None):
                raise ValueError("self has value while other does not")

            new_crow = bm.array([0],dtype=bm.int64)
            new_col = bm.array([],dtype=bm.int64)
            new_values = bm.array([],dtype=bm.int64)

            for i in range(0, self._crow.shape[0]-1): 
                indices1 = self._col[self._crow[i]:self._crow[i+1]]
                indices2 = other._col[other._crow[i]:other._crow[i+1]]
                col, inverse_indices = bm.unique(bm.concat((indices1,indices2)), return_inverse=True)

                if self._values is None:
                    new_values = None
                else:
                    value1 = self._values[self._crow[i]:self._crow[i+1]]
                    value2 = other._values[other._crow[i]:other._crow[i+1]]
                    val =bm.concat((value1,alpha*value2))
                    values = bm.zeros(col.shape[0],dtype=val.dtype)
                    values = bm.index_add(values, inverse_indices, val, axis=-1)
                    new_values = bm.concat((new_values,values))
                new_crow = bm.concat((new_crow,bm.tensor([len(col)+new_crow[-1]])))
                new_col = bm.concat((new_col,col))

            return CSRTensor(new_crow, new_col,new_values ,self.sparse_shape)

        elif isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            output = other * alpha
            context = bm.context(output)
            output = output.reshape(self.dense_shape + (prod(self._spshape),))

            count = self._crow[1:] - self._crow[:-1]
            nrow = self._crow.shape[0] - 1
            row = bm.repeat(bm.arange(nrow), count)
            indices = bm.stack([row, self._col], axis=0)
            flattened = flatten_indices(indices, self._spshape)[0]

            if self._values is None:
                src = bm.ones((1,) * (self.dense_ndim + 1), **context)
                src = bm.broadcast_to(src, self.dense_ndim + (self.nnz,))
            else:
                src = self._values
            output = bm.index_add(output, flattened, src, axis=-1)

            return output.reshape(self.shape)

        elif isinstance(other, (int, float)):
            new_values = self._values + alpha * other
            return CSRTensor(bm.copy(self._crow), bm.copy(self._col),new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in addition")

    def mul(self, other: Union[Number, 'CSRTensor', TensorLike]) -> 'CSRTensor':
        """Element-wise multiplication.
        The result CSR tensor will share the same indices with
        the original if `other` is a number or a dense tensor.
        """
        if isinstance(other, CSRTensor):
            pass

        elif isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            new_values = bm.copy(other[self.nonzero_slice])

            if self._values is not None:
                bm.multiply(self._values, new_values, out=new_values)

            return CSRTensor(self._crow, self._col,new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            if self._values is None:
                raise ValueError("Cannot multiply CSRTensor without value with scalar")
            new_values = self._values * other

            return CSRTensor(self._crow,self._col, new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in multiplication")

    def div(self, other: Union[Number, TensorLike]) -> 'CSRTensor':
        """Element-wise division.
        The result CSR tensor will share the same indices with
        the original if `other` is a number or a dense tensor.
        """
        if self._values is None:
                raise ValueError("Cannot divide CSRTensor without value")

        if isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            new_values = bm.copy(other[self.nonzero_slice])
  
            bm.divide(self._values, new_values, out=new_values)
            return CSRTensor(self._crow,self._col,new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            new_values = self._values / other
            return CSRTensor(self._indices, new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in division")

    def pow(self, other: Union[TensorLike, Number]) -> 'CSRTensor':
        """Element-wise power of CSRTensor.
        The result CSR tensor will share the same indices with
        the original if `other` is a number or a dense tensor.
        """
        if self._values is None:
            raise ValueError("Cannot power CSRTensor without value with tensor")

        if isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            new_values = bm.copy(other[self.nonzero_slice])

            new_values = bm.power(self._values, new_values)
            return CSRTensor(self._crow, self._col,new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            new_values = self._values ** other
            return CSRTensor(self._indices, new_values, self.sparse_shape)

        else:
            raise TypeError(f'Unsupported type {type(other).__name__} in power')

    @overload
    def matmul(self, other: 'CSRTensor') -> 'CSRTensor': ...
    @overload
    def matmul(self, other: TensorLike) -> TensorLike: ...
    def matmul(self, other: Union['CSRTensor', TensorLike]):
        """Matrix-multiply this CSRTensor with another tensor.

        Parameters:
            other (CSRTensor | Tensor): A 1-D tensor for matrix-vector multiply,
                or a 2-D tensor for matrix-matrix multiply.
                Batched matrix-matrix multiply is available for dimensions
                (*B, M, K) and (*B, K, N). *B means any number of batch dimensions.

        Raises:
            TypeError: If the type of `other` is not supported for matmul.

        Returns:
            out (CSRTensor | Tensor): A new CSRTensor if `other` is a CSRTensor,\
            or a Tensor if `other` is a dense tensor.
        """
        if isinstance(other, CSRTensor):
            if (self.values() is None) or (other.values() is None):
                raise ValueError("Matrix multiplication between CSRTensor without "
                                 "value is not implemented now")
            crow, col,values, spshape = spspmm_csr(
                self._crow,self._col ,self._values, self.sparse_shape,
                other._crow, other._col,other._values, other.sparse_shape,
            )
            return CSRTensor(crow, col,values, spshape)

        elif isinstance(other, TensorLike):
            if self.values() is None:
                raise ValueError()
            try:
                return bm.csr_spmm(self._crow, self._col, self._values, self._spshape, other)
            except (AttributeError, NotImplementedError):
                pass

            return spmm_csr(self._crow, self._col,self._values,self.sparse_shape, other)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in matmul")
