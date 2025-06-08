
from typing import Optional, Union, overload, Dict, Sequence, Any, TypeVar, Type
from math import prod

from ..backend import TensorLike, Number, Size
from ..backend import backend_manager as bm
from .utils import _dense_ndim, _dense_shape

_Self = TypeVar('_Self', bound='SparseTensor')


class SparseTensor():
    _values: Optional[TensorLike]
    _spshape: Size

    ### 1. Data Fetching ###
    # NOTE: These properties should be supported by the indices system
    # in all subclasses.
    @property
    def itype(self): raise NotImplementedError
    @property
    def nnz(self) -> int: raise NotImplementedError

    def values_context(self) -> Dict[str, Any]:
        if self._values is None:
            return {}
        return bm.context(self._values)

    @property
    def ftype(self): return None if self._values is None else self._values.dtype

    @property
    def dtype(self): return None if self._values is None else self._values.dtype

    @property
    def shape(self): return self.dense_shape + self.sparse_shape
    @property
    def dense_shape(self): return _dense_shape(self._values)
    @property
    def sparse_shape(self): return self._spshape

    @property
    def ndim(self): return self.dense_ndim + self.sparse_ndim
    @property
    def dense_ndim(self): return _dense_ndim(self._values)
    @property
    def sparse_ndim(self): return len(self._spshape)

    def size(self, dim: Optional[int]=None) -> int:
        """Number of elements as a dense array."""
        if dim is None:
            return prod(self.shape)
        else:
            return self.shape[dim]

    ### 2. Data Type & Device Management ###
    def astype(self: _Self, dtype=None, /, *, copy=True) -> _Self:
        raise NotImplementedError

    def device_put(self: _Self, device=None, /) -> _Self:
        raise NotImplementedError

    ### 3. Format Conversion ###
    def to_dense(self, *, fill_value: Union[Number, bool] = 1, dtype=None) -> TensorLike:
        """Convert to a dense tensor and return as a new object.

        Parameters:
            fill_value (int | float, optional): The value to fill the dense tensor with
                when `self.values()` is None.
            dtype (dtype, optional): The scalar type of elements. This is useful
                when `self.values()` is None. Defaults to float64.

        Returns:
            Tensor: The dense tensor.
        """
        raise NotImplementedError

    def toarray(self, *, fill_value: Union[Number, bool] = 1, dtype=None) -> TensorLike:
        return self.to_dense(fill_value=fill_value, dtype=dtype)

    def tocoo(self, *, copy=False):
        raise NotImplementedError

    def tocsr(self, *, copy=False):
        raise NotImplementedError

    ### 4. Object Conversion ###
    def to_scipy(self):
        raise NotImplementedError

    def from_scipy(cls, mat, /):
        raise NotImplementedError

    ### 5. Manipulation ###
    def copy(self: _Self) -> _Self:
        raise NotImplementedError

    def coalesce(self: _Self, accumulate: bool=True) -> _Self:
        """Sum the duplicated indices and return as a new sparse tensor.
        Returns self if the indices are already coalesced.

        Parameters:
            accumulate (bool, optional): Whether to count the occurrences of indices\
            as new values when `self.values` is None. Defaults to True.

        Returns:
            SparseTensor: coalesced sparse tensor.
        """
        raise NotImplementedError

    def reshape(self: _Self, *shape) -> _Self:
        raise NotImplementedError

    def ravel(self: _Self) -> _Self:
        """Return a view with flatten indices on sparse dimensions.

        Returns:
            SparseTensor: A flatten tensor, shaped (*dense_shape, -1).
        """
        return self.reshape(-1)

    def flatten(self: _Self) -> _Self:
        """Return a copy with flatten indices on sparse dimensions.

        Returns:
            SparseTensor: A flatten tensor, shaped (*dense_shape, -1).
        """
        raise NotImplementedError

    @property
    def T(self: _Self) -> _Self:
        raise NotImplementedError

    def tril(self, k: int=0) -> _Self:
        raise NotImplementedError

    def triu(self, k: int=0) -> _Self:
        raise NotImplementedError

    @classmethod
    def concat(cls: Type[_Self], tensors: Sequence[_Self], /, *, axis: int=0) -> _Self:
        raise NotImplementedError

    ### 6. Arithmetic Operations ###
    def neg(self: _Self) -> _Self:
        raise NotImplementedError

    @overload
    def add(self: _Self, other: Union[Number, _Self], alpha: Number=1) -> _Self: ...
    @overload
    def add(self: _Self, other: TensorLike, alpha: Number=1) -> TensorLike: ...
    def add(self, other, alpha: Number=1):
        raise NotImplementedError

    def mul(self: _Self, other: Union[Number, _Self, TensorLike]) -> _Self:
        raise NotImplementedError

    def div(self: _Self, other: Union[Number, TensorLike]) -> _Self:
        raise NotImplementedError

    def pow(self: _Self, other: Union[TensorLike, Number]) -> _Self:
        raise NotImplementedError

    @overload
    def matmul(self: _Self, other: _Self) -> _Self: ...
    @overload
    def matmul(self: _Self, other: TensorLike) -> TensorLike: ...
    def matmul(self, other):
        raise NotImplementedError

    def __pos__(self: _Self): return self
    def __neg__(self: _Self): return self.neg()

    @overload
    def __add__(self: _Self, other: Union[_Self, Number]) -> _Self: ...
    @overload
    def __add__(self: _Self, other: TensorLike) -> TensorLike: ...
    def __add__(self, other):
        return self.add(other)

    @overload
    def __radd__(self: _Self, other: Union[_Self, Number]) -> _Self: ...
    @overload
    def __radd__(self: _Self, other: TensorLike) -> TensorLike: ...
    def __radd__(self, other):
        return self.add(other)

    @overload
    def __sub__(self: _Self, other: Union[_Self, Number]) -> _Self: ...
    @overload
    def __sub__(self: _Self, other: TensorLike) -> TensorLike: ...
    def __sub__(self, other):
        return self.add(-other)

    @overload
    def __rsub__(self: _Self, other: Union[_Self, Number]) -> _Self: ...
    @overload
    def __rsub__(self: _Self, other: TensorLike) -> TensorLike: ...
    def __rsub__(self, other):
        return self.neg().add(other)

    def __mul__(self: _Self, other: Union[_Self, TensorLike, Number]) -> _Self:
        return self.mul(other)
    __rmul__ = __mul__

    def __truediv__(self: _Self, other: Union[TensorLike, Number]) -> _Self:
        return self.div(other)

    def __pow__(self: _Self, other: Union[TensorLike, Number]) -> _Self:
        return self.pow(other)

    @overload
    def __matmul__(self: _Self, other: _Self) -> _Self: ...
    @overload
    def __matmul__(self: _Self, other: TensorLike) -> TensorLike: ...
    def __matmul__(self, other):
        return self.matmul(other)
