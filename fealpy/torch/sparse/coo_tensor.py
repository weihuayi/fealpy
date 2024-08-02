
from typing import Optional, Union, overload, List

import torch

from ..typing import Tensor, Number, Size, _dtype, _device
from .utils import (
    _dense_ndim, _dense_shape, _flatten_indices,
    check_shape_match, check_spshape_match
)
from ._spspmm import spspmm_coo
from ._spmm import spmm_coo


class COOTensor():
    def __init__(self, indices: Tensor, values: Optional[Tensor],
                 spshape: Optional[Size]=None, *,
                 dtype: Optional[_dtype]=None,
                 device: Union[str, _device, None]=None,
                 is_coalesced: Optional[bool]=None):
        """
        Initialize COO format sparse tensor.

        Parameters:
            indices (Tensor): indices of non-zero elements, shaped (D, N).
                Where D is the number of sparse dimension, and N is the number
                of non-zeros (nnz).
            values (Tensor | None): non-zero elements, shaped (..., N).
            spshape (Size | None, optional): shape in the sparse dimensions.
        """
        if device is not None:
            indices = indices.to(device)
            if values is not None:
                values = values.to(device)
        else:
            if values is not None and values.device != indices.device:
                raise ValueError("values and indices must be on the same device")

        if values is not None and dtype is not None:
            values = values.to(dtype=dtype)

        self._indices = indices
        self._values = values
        self.is_coalesced = is_coalesced

        if spshape is None:
            self._spshape = Size((torch.max(indices, dim=1)[0] + 1,))
        else:
            self._spshape = Size(spshape)

        self._check(indices, values, spshape)

    def _check(self, indices: Tensor, values: Optional[Tensor], spshape: Size):
        if not isinstance(indices, Tensor):
            raise TypeError(f"indices must be a Tensor, but got {type(indices)}")
        if indices.ndim != 2:
            raise ValueError(f"indices must be a 2D tensor, but got {indices.ndim}D")

        # total ndim should be equal to sparse_ndim + dense_dim
        if len(spshape) != indices.shape[0]:
            raise ValueError(f"size must have length {indices.shape[0]}, "
                             f"but got {len(spshape)}")

        if isinstance(values, Tensor):
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

    @overload
    def size(self) -> Size: ...
    @overload
    def size(self, dim: int) -> int: ...
    def size(self, dim: Optional[int]=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @property
    def device(self): return self._indices.device
    @property
    def dtype(self):
        if self._values is None:
            return self._indices.dtype
        else:
            return self._values.dtype

    @property
    def shape(self): return Size(self.dense_shape + self.sparse_shape)
    @property
    def dense_shape(self): return _dense_shape(self._values)
    @property
    def sparse_shape(self): return self._spshape

    @property
    def ndim(self): return self.dense_ndim + self.sparse_ndim
    @property
    def dense_ndim(self): return _dense_ndim(self._values)
    @property
    def sparse_ndim(self): return self._indices.shape[0]
    @property
    def nnz(self): return self._indices.shape[1]

    def indices(self) -> Tensor:
        """Return the indices of the non-zero elements."""
        return self._indices

    def values(self) -> Optional[Tensor]:
        """Return the non-zero elements."""
        return self._values

    def to_dense(self) -> Tensor:
        """Convert the COO tensor to a dense tensor and return as a new object."""
        if not self.is_coalesced:
            raise ValueError("indices must be coalesced before calling to_dense()")

        kwargs = {'dtype': self.dtype, 'device': self.device}
        dense_tensor = torch.zeros(self.shape, **kwargs)
        slicing = [self._indices[i] for i in range(self.sparse_ndim)]

        if self._values is None:
            dense_tensor[slicing] = 1
        else:
            slicing = [slice(None),] * self.dense_ndim + slicing
            dense_tensor[slicing] = self._values

        return dense_tensor

    toarray = to_dense

    def to(self, device: Union[_device, str, None]=None, non_blocking: bool=False) -> Tensor:
        """Return a new COOTensor object with the same indices and values on the
        specified device."""
        pass

    def coalesce(self, accumulate: bool=True) -> 'COOTensor':
        """Merge duplicate indices and return as a new COOTensor object.
        Returns self if the indices are already coalesced.

        Parameters:
            accumulate (bool, optional): Whether to count the occurrences of indices\
            as new values when `self.values` is None. Defaults to True.

        Returns:
            COOTensor: coalesced COO tensor.
        """
        if self.is_coalesced:
            return self

        kwargs = {'dtype': self.dtype, 'device': self.device}

        unique_indices, inverse_indices = torch.unique(
            self._indices, return_inverse=True, dim=1
        )

        if self._values is not None:
            dense_ndim = self.dense_ndim
            value_shape = self.dense_shape + (unique_indices.size(-1), )
            new_values = torch.zeros(value_shape, **kwargs)
            inverse_indices = inverse_indices[(None,) * dense_ndim + (slice(None),)]
            inverse_indices = inverse_indices.broadcast_to(self._values.shape)
            new_values.scatter_add_(-1, inverse_indices, self._values)

            return COOTensor(
                unique_indices, new_values, self.sparse_shape, is_coalesced=True
            )

        else:
            if accumulate:
                ones = torch.ones((self.nnz, ), **kwargs)
                new_values = torch.zeros((unique_indices.size(-1), ), **kwargs)
                new_values.scatter_add_(-1, inverse_indices, ones)
            else:
                new_values = None

            return COOTensor(
                unique_indices, new_values, self.sparse_shape, is_coalesced=True
            )

    @overload
    def reshape(self, shape: Size, /) -> 'COOTensor': ...
    @overload
    def reshape(self, *shape: int) -> 'COOTensor': ...
    def reshape(self, *shape) -> 'COOTensor':
        pass

    def ravel(self) -> 'COOTensor':
        """Return a view with flatten indices on sparse dimensions.

        Returns:
            COOTensor: A flatten COO tensor, shaped (*dense_shape, 1).
        """
        spshape = self.sparse_shape
        new_indices = _flatten_indices(self._indices, spshape)
        return COOTensor(new_indices, self._values, (spshape.numel(),))

    def flatten(self) -> 'COOTensor':
        """Return a copy with flatten indices on sparse dimensions.

        Returns:
            COOTensor: A flatten COO tensor, shaped (*dense_shape, 1).
        """
        spshape = self.sparse_shape
        new_indices = _flatten_indices(self._indices, spshape)
        if self._values is None:
            values = None
        else:
            values = self._values.clone()
        return COOTensor(new_indices, values, (spshape.numel(),))

    @overload
    def add(self, other: Union[Number, 'COOTensor'], alpha: float=1.0) -> 'COOTensor': ...
    @overload
    def add(self, other: Tensor, alpha: float=1.0) -> Tensor: ...
    def add(self, other: Union[Number, 'COOTensor', Tensor], alpha: float=1.0) -> Union['COOTensor', Tensor]:
        """Adds another tensor or scalar to this COOTensor, with an optional scaling factor.

        Parameters:
            other (Number | COOTensor | Tensor): The tensor or scalar to be added.\n
            alpha (float, optional): The scaling factor for the other tensor. Defaults to 1.0.

        Raises:
            TypeError: If the type of `other` is not supported for addition.\n
            ValueError: If the shapes of `self` and `other` are not compatible.\n
            ValueError: If one has value and another does not.

        Returns:
            COOTensor | Tensor: A new COOTensor if `other` is a COOTensor,\
            or a Tensor if `other` is a dense tensor.
        """
        if isinstance(other, COOTensor):
            check_shape_match(self.shape, other.shape)
            check_spshape_match(self.sparse_shape, other.sparse_shape)
            new_indices = torch.cat((self._indices, other._indices), dim=1)
            if self._values is None:
                if other._values is None:
                    new_values = None
                else:
                    raise ValueError("self has no value while other does")
            else:
                if other._values is None:
                    raise ValueError("self has value while other does not")
                new_values = torch.cat((self._values, other._values*alpha), dim=-1)
            return COOTensor(new_indices, new_values, self.sparse_shape)

        elif isinstance(other, Tensor):
            check_shape_match(self.shape, other.shape)
            output = other * alpha
            slicing = [self._indices[i] for i in range(self.sparse_ndim)]
            slicing = [slice(None),] * self.dense_ndim + slicing
            if self._values is None:
                output[slicing] += 1.
            else:
                output[slicing] += self._values
            return output

        elif isinstance(other, (int, float)):
            new_values = self._values + alpha * other
            return COOTensor(self._indices.clone(), new_values, self.sparse_shape)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in addition")

    def mul(self, other: Union[Number, 'COOTensor', Tensor]) -> 'COOTensor': # TODO: finish this
        if isinstance(other, COOTensor):
            pass
        elif isinstance(other, Tensor):
            pass
        elif isinstance(other, (int, float)):
            if self._values is None:
                raise ValueError("Cannot multiply COOTensor without value with scalar")
            new_values = self._values * other
            return COOTensor(self._indices.clone(), new_values, self.sparse_shape)
        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in multiplication")

    def div(self, other: Union[Number, Tensor]) -> 'COOTensor':
        pass

    def inner(self, other: Tensor, dims: List[int]) -> 'COOTensor':
        pass

    @overload
    def matmul(self, other: 'COOTensor') -> 'COOTensor': ...
    @overload
    def matmul(self, other: Tensor) -> Tensor: ...
    def matmul(self, other: Union['COOTensor', Tensor]):
        """Matrix-multiply this COOTensor with another tensor.

        Parameters:
            other (COOTensor | Tensor): A 2-D tensor.

        Raises:
            TypeError: If the type of `other` is not supported for matmul.

        Returns:
            COOTensor | Tensor: A new COOTensor if `other` is a COOTensor,\
            or a Tensor if `other` is a dense tensor.
        """
        if isinstance(other, COOTensor):
            if (self.values() is None) or (other.values() is None):
                raise ValueError("Matrix multiplication between COOTensor without "
                                 "value is not implemented now")
            indices, values, spshape = spspmm_coo(
                self.indices(), self.values(), self.sparse_shape,
                other.indices(), other.values(), other.sparse_shape,
            )
            return COOTensor(indices, values, spshape).coalesce()

        elif isinstance(other, Tensor):
            if self.values() is None:
                raise ValueError()
            return spmm_coo(self.indices(), self.values(), self.sparse_shape, other)

        else:
            raise TypeError(f"Unsupported type {type(other).__name__} in matmul")
