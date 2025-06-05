
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
from .coo_tensor import COOTensor

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
    def itype(self): return self._col.dtype

    @property
    def nnz(self): return self._col.shape[-1]

    @property
    def crow(self) -> TensorLike:
        """Return the row location of non-zero elements."""
        return self._crow

    @property
    def col(self): return self._col

    @property
    def values(self) -> Optional[TensorLike]:
        """Return the non-zero elements"""
        return self._values

    @property
    def row(self):
        count = self._crow[1:] - self._crow[:-1]
        nrow = self._crow.shape[0] - 1
        kargs = bm.context(self._crow)
        return bm.repeat(bm.arange(nrow, **kargs), count)

    @property
    def indptr(self): return self._crow # scipy convention

    @property
    def indices(self): return self._col # scipy convention

    @property
    def data(self): return self._values # scipy convention

    @property
    def nonzero_slice(self) -> Tuple[Union[slice, TensorLike]]:
        return self.row, self._col

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
    def to_dense(self, *, fill_value: Union[Number, bool] = 1, dtype=None) -> TensorLike:
        if self.values is None:
            dtype = bm.float64 if (dtype is None) else dtype
            context = {"dtype": dtype, "device": bm.get_device(self.indices)}
            src = bm.full((1,) * (self.dense_ndim + 1), fill_value, **context)
            src = bm.broadcast_to(src, self.dense_shape + (self.nnz,))
        else:
            src = self.values if (dtype is None) else bm.astype(self.values, dtype)
            context = {"dtype": src.dtype, "device": bm.get_device(src)}

        index_context = {'dtype': self._crow.dtype, 'device': bm.get_device(self._crow)}

        count = self._crow[1:] - self._crow[:-1]
        nrow = self._crow.shape[0] - 1
        row = bm.repeat(bm.arange(nrow, **index_context), count)
        indices = bm.stack([row, self._col], axis=0)

        dense_tensor = bm.zeros(self.dense_shape + (prod(self._spshape),), **context)
        flattened = flatten_indices(indices, self._spshape)[0]
        dense_tensor = bm.index_add(dense_tensor, flattened, src, axis=-1)

        return dense_tensor.reshape(self.shape)

    def tocoo(self, *, copy=False):
        """
        """
        from .coo_tensor import COOTensor
        indices = bm.stack(self.nonzero_slice, axis=0)
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
        """
        """
        A = self.tocoo()
        return A.T.tocsr()

    def partial(self, index: Union[TensorLike, slice]):
        crow = self.crow
        ZERO = bm.zeros([1], dtype=crow.dtype, device=bm.get_device(crow))
        new_col = bm.copy(self.col[..., index])
        is_selected = bm.zeros((self.nnz,), dtype=bm.bool, device=bm.get_device(new_col))
        is_selected = bm.set_at(is_selected, index, True)
        selected_cum = bm.concat([ZERO, bm.cumsum(is_selected, axis=0)], axis=0)
        new_nnz_per_row = selected_cum[crow[1:]] - selected_cum[crow[:-1]]
        new_crow = bm.concat([ZERO, bm.cumsum(new_nnz_per_row, axis=0)], axis=0)

        new_values = self.values

        if new_values is not None:
            new_values = bm.copy(new_values[..., index])

        return CSRTensor(new_crow, new_col, new_values, self._spshape)

    def tril(self, k: int = 0) -> 'CSRTensor':
        tril_loc = (self.row + k) >= self.col
        return self.partial(tril_loc)

    def triu(self, k: int = 0) -> 'CSRTensor':
        tril_loc = (self.col - k) >= self.row
        return self.partial(tril_loc)

    def sum(self, axis=0):
        """
        """
        kargs = bm.context(self._values)
        if axis == 0: # the sum of row
            return self@bm.ones(self._spshape[1], **kargs)
        elif axis == 1: # the sum of column
            r = bm.zeros(self._spshape[1], **kargs)
            r = bm.index_add(r, self._col, self._values)
            return r

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
        self_indices = bm.stack(self.nonzero_slice, axis=0)
        if isinstance(other, CSRTensor):
            other_indices = bm.stack(other.nonzero_slice, axis=0)
            check_shape_match(self.shape, other.shape)
            check_spshape_match(self.sparse_shape, other.sparse_shape)
            
            new_indices = bm.concat((self_indices, other_indices), axis=1)
            context = bm.context(new_indices)
            if self._values is None:
                if other._values is None:
                    self._values = bm.zeros((self._crow[-1], ), **context) + 1.0
                    other._values = bm.zeros((other._crow[-1], ), **context) + 1.0
                else:
                    self._values = bm.zeros((self._crow[-1], ), **context) + 1.0
            else:
                if other._values is None:
                    other._values = bm.zeros((other._crow[-1], ), **context) + 1.0
            new_values = bm.concat((self._values, other._values*alpha), axis=-1)
            return COOTensor(new_indices, new_values, self.sparse_shape).tocsr()

        elif isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)
            output = other * alpha
            context = bm.context(output)
            output = output.reshape(self.dense_shape + (prod(self._spshape),))
            flattened = flatten_indices(self_indices, self._spshape)[0]

            if self._values is None:
                src = bm.ones((1,) * (self.dense_ndim + 1), **context)
                src = bm.broadcast_to(src, self.dense_ndim + (self.nnz,))
            else:
                src = self._values
            output = bm.index_add(output, flattened, src, axis=-1)

            return output.reshape(self.shape)

        elif isinstance(other, (int, float)):
            new_values = self._values + alpha * other
            return CSRTensor(bm.copy(self._crow), bm.copy(self._col), new_values, self.sparse_shape)

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

            return CSRTensor(self._crow, self._col, new_values, self.sparse_shape)

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
            if len(other.shape) == 1: #TODO: deal with case self.shape[0] == self.shape[1]
                if other.shape[0] == self.shape[0]:
                    other = bm.broadcast_to(other[:, None], self.shape)
                elif other.shape[0] == self.shape[1]:
                    other = bm.broadcast_to(other[None, :], self.shape)
            check_shape_match(other.shape, self.shape)
            new_values = bm.copy(other[self.nonzero_slice])
  
            bm.divide(self._values, new_values, out=new_values)
            return CSRTensor(self._crow, self._col, new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            new_values = self._values / other
            return CSRTensor(self._crow, self._col, new_values, self.sparse_shape)

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
            return CSRTensor(self._crow, self._col, new_values, self.sparse_shape)

        elif isinstance(other, (int, float)):
            new_values = self._values ** other
            return CSRTensor(self._crow, self._col, new_values, self.sparse_shape)

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
            if (self.values is None) or (other.values is None):
                raise ValueError("Matrix multiplication between CSRTensor without "
                                 "value is not implemented now")
            if hasattr(bm, 'csr_spspmm'):
                crow, col, values, spshape = bm.csr_spspmm(
                    self.crow, self.col, self.values, self.sparse_shape,
                    other.crow, other.col, other.values, other.sparse_shape
                )
            else:
                crow, col,values, spshape = spspmm_csr(
                    self._crow,self._col ,self._values, self.sparse_shape,
                    other._crow, other._col,other._values, other.sparse_shape,
                )
            return CSRTensor(crow, col, values, spshape)

        elif isinstance(other, TensorLike):
            if self.values is None:
                raise ValueError()
            if hasattr(bm, 'csr_spmm'):
                return bm.csr_spmm(self._crow, self._col, self._values, self._spshape, other)
            else:
                return spmm_csr(self._crow, self._col,self._values,self.sparse_shape, other)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in matmul")


    def find(self):
        """
        Find the non-zero entries in the sparse matrix..

        Returns:
                - row indices of non-zero values.
                - column indices of non-zero values.
                - non-zero values themselves.
        """
        nz_mask = self.values != 0
        return self.row[nz_mask], self.col[nz_mask], self.values[nz_mask]

    def diags(self) -> 'CSRTensor':
        """
        Extract the diagonal elements from the sparse matrix.

        Returns:
            CSRTensor: A new CSRTensor object containing the diagonal values.
        """
        diags_loc = (self.row) == self.col
        return self.partial(diags_loc)

    def col_min(self):
        """
        Compute the minimum values in each column of the sparse matrix.

        Returns:
            Tensor: A tensor containing the minimum values for each column.
        """
        M = bm.zeros(self._spshape[1], dtype=self._values.dtype)
        bm.minimum.at(M, self._col, self._values)
        
        return M

    def __getitem__(self, index):
        if isinstance(index, Tuple):
            crow_index, col_index = index 
        else:
            crow_index = index
            col_index = None

        if col_index is not None:
            if isinstance(col_index, slice):
                start = col_index.start if col_index.start is not None else 0
                stop = col_index.stop if col_index.stop is not None else self._spshape[1]
                step = col_index.step if col_index.step is not None else 1
                new_shape = (stop - start + step - 1) // step
                col_index = bm.arange(start, stop, step)
                if new_shape == self._spshape[1]:
                    new_crow = self._crow
                    new_col = self._col
                    new_values = self._values
                    new_col_shape = self._spshape[1] 
            elif isinstance(col_index, (List, TensorLike)):
                new_shape = len(col_index)
            elif isinstance(col_index, int):
                new_shape = 1
            else:
                raise TypeError(f'index must be a slice or int, but got {type(index)}')

            kwargs = bm.context(self._col)
            nrz = self.crow[1:] - self.crow[:-1]
            isfindnode = bm.zeros((self._spshape[1],), **kwargs)
            isfindnode = bm.add_at(isfindnode, col_index, 1) == 1
            row = bm.repeat(bm.arange(self._spshape[0]), nrz)
            new_row = row[isfindnode[self._col]]
            new_row = bm.concat((new_row, [self._spshape[0] - 1]))
            new_crow = bm.concat(([0], bm.cumsum(bm.bincount(new_row))))
            new_crow[-1] = new_crow[-1] - 1
            new_values = self._values[isfindnode[self._col]]
            if isinstance(col_index, int):
                new_col = bm.zeros((self._col[isfindnode[self._col]].shape[0],), dtype=bm.int64)
            else:
                a = bm.searchsorted(col_index, self._col[isfindnode[self._col]]) 
                new_col = bm.arange(new_shape)[a]
            new_col_shape = new_shape
        else:
            new_crow = self._crow
            new_col = self._col
            new_values = self._values
            new_col_shape = self._spshape[1]

        if isinstance(crow_index, slice):
            start = crow_index.start if crow_index.start is not None else 0
            stop = crow_index.stop if crow_index.stop is not None else new_crow.shape[0] - 1
            step = crow_index.step if crow_index.step is not None else 1
            new_row_shape = (stop - start + step - 1) // step
            if new_row_shape == new_crow.shape[0] - 1:
                return CSRTensor(new_crow, new_col, new_values, spshape=(new_row_shape, new_col_shape)) 
        elif isinstance(crow_index, (List, TensorLike)):
            new_row_shape = len(crow_index)
        elif isinstance(crow_index, int):
            new_row_shape = 1
        else:
            raise TypeError(f'index must be a slice or int, but got {type(index)}')

        kwargs = bm.context(self.crow)
        nrz = new_crow[1:] - new_crow[:-  1]
        isfindnode = bm.zeros((new_crow.shape[0] - 1,), **kwargs)
        isfindnode = bm.add_at(isfindnode, crow_index, 1)
        findnode = bm.repeat(isfindnode == 1, nrz) 
        new_col = new_col[findnode]
        new_values = new_values[findnode]
        new_crow = bm.concat(([0], bm.cumsum(nrz[isfindnode == 1])))
        
        return CSRTensor(new_crow, new_col, new_values, spshape=(new_row_shape, new_col_shape))