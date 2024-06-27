
from typing import Optional, Union, overload, List

import torch

from .utils import _dense_ndim, _dense_shape


_Size = torch.Size
_dtype = torch.dtype
_device = torch.device
Tensor = torch.Tensor
Number = Union[int, float]


class CSRTensor():
    def __init__(self, crow: Tensor, col: Tensor, values: Optional[Tensor],
                 spshape: Optional[_Size]=None, *,
                 dtype: Optional[_dtype]=None,
                 device: Union[str, _device, None]=None) -> None:
        """Initializes CSR format sparse tensor.

        Parameters:
            crow (Tensor): _description_
            col (Tensor): _description_
            values (Tensor | None): _description_
            spshape (Size | None, optional): _description_
        """
        self.crow = crow
        self.col = col
        self.values = values

        if spshape is None:
            nrow = crow.size(0) - 1
            ncol = col.max().item() + 1
            self._spshape = _Size((nrow, ncol))
        else:
            self._spshape = _Size(spshape)

        self._check(crow, col, values, spshape)

    def _check(self, crow: Tensor, col: Tensor, values: Optional[Tensor], spshape: _Size):
        if crow.ndim != 1:
            raise ValueError(f"crow must be a 1-D tensor, but got {crow.ndim}")
        if col.ndim != 1:
            raise ValueError(f"col must be a 1-D tensor, but got {col.ndim}")
        if len(spshape) != 2:
                raise ValueError(f"spshape must be a 2-tuple, but got {spshape}")

        if spshape[0] != crow.size(0) - 1:
            raise ValueError(f"crow.size(0) - 1 must be equal to spshape[0], "
                             f"but got {crow.size(0) - 1} and {spshape[0]}")

        if isinstance(values, Tensor):
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
        return f"CSRTensor(crow={self.crow}, col={self.col}, "\
               + f"values={self.values}, shape={self.shape})"

    @overload
    def size(self) -> _Size: ...
    @overload
    def size(self, dim: int) -> int: ...
    def size(self, dim: Optional[int]=None):
        if dim is None:
            return self.shape
        else:
            return self.shape[dim]

    @property
    def device(self): return self.crow.device
    @property
    def dtype(self):
        if self.values is None:
            return self.crow.dtype
        else:
            return self.values.dtype

    @property
    def shape(self): return _Size(self.dense_shape + self.sparse_shape)
    @property
    def dense_shape(self): return _dense_shape(self.values)
    @property
    def sparse_shape(self): return self._spshape

    @property
    def ndim(self): return self.dense_ndim + self.sparse_ndim
    @property
    def dense_ndim(self): return _dense_ndim(self.values)
    @property
    def sparse_ndim(self): return 2
    @property
    def nnz(self): return self.col.shape[1]

    def to_dense(self) -> Tensor:
        """Convert the CSRTensor to a dense tensor and return as a new object."""
        kwargs = {'dtype': self.dtype, 'device': self.device}
        dense_tensor = torch.zeros(self.shape, **kwargs)

        for i in range(1, self.crow.shape[0]):
            start = self.crow[i - 1]
            end = self.crow[i]
            dense_tensor[..., i - 1, self.col[start:end]] = self.values[..., start:end]

        return dense_tensor

    def to(self, device: Union[_device, str, None]=None, non_blocking: bool=False):
        pass
