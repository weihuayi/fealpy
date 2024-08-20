
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
            crow (Tensor): _description_
            col (Tensor): _description_
            values (Tensor | None): _description_
            spshape (Size | None, optional): _description_
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

    @property
    def itype(self): return self._crow.dtype

    @property
    def nnz(self): return self._col.shape[1]

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

    def col(self) -> TensorLike:
        """Return the column of non-zero elements."""
        return self._col

    def values(self) -> Optional[TensorLike]:
        """Return the non-zero elements"""
        return self._values

    def to_dense(self, *, fill_value: Number=1.0, **kwargs) -> TensorLike:
        """Convert the CSRTensor to a dense tensor and return as a new object.

        Parameters:
            fill_value (int | float, optional): The value to fill the dense tensor with
                when `self.values()` is None.

        Returns:
            Tensor: The dense tensor.
        """
        context = self.values_context()
        context.update(kwargs)
        dense_tensor = bm.zeros(self.shape, **context)

        for i in range(1, self._crow.shape[0]):
            start = self._crow[i - 1]
            end = self._crow[i]
            val = fill_value if (self._values is None) else self._values[..., start:end]
            dense_tensor[..., i - 1, self._col[start:end]] = val

        return dense_tensor

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

    def copy(self):
        return CSRTensor(bm.copy(self._crow), bm.copy(self._col),
                         bm.copy(self._values), self._spshape)

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
                    values = bm.zeros(col.shape[0],dtype=value2.dtype)
                    values = bm.index_add(values, inverse_indices, bm.concat((value1,alpha*value2)), axis=-1)
                    new_values = bm.concat((new_values,values))
                new_crow = bm.concat((new_crow,bm.tensor([len(col)+new_crow[-1]])))
                new_col = bm.concat((new_col,col))

            return CSRTensor(new_crow, new_col,new_values ,self.sparse_shape)

        elif isinstance(other, TensorLike):
            check_shape_match(self.shape, other.shape)

            output = other * alpha
            context = bm.context(output)

            output = output.reshape(self.dense_shape + (prod(self._spshape),))

            indices1 = bm.zeros([2,len(self._col)],dtype=bm.int64)
            for x in range(self.shape[-1]):
                indices1[0,self._crow[x]:self._crow[x+1]] = x
            indices1[1,:] = self._col

            flattened = flatten_indices(indices1, self._spshape)[0]

            if self._values is None:
                src = bm.ones((1,) * (self.dense_ndim + 1), **context)
                src = bm.broadcast_to(src, self.dense_ndim + (self.nnz,))
            else:
                src = self._values
            bm.index_add(output, -1, flattened, src)

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
            return spmm_csr(self._crow, self._col,self._values,self.sparse_shape, other)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in matmul")
