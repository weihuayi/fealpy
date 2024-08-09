
from typing import Optional, Union, overload, Dict, Any, TypeVar
from math import prod

from ..backend import TensorLike, Number, Size
from ..backend import backend_manager as bm
from .utils import _dense_ndim, _dense_shape

_Self = TypeVar('_Self', bound='SparseTensor')


class SparseTensor():
    _values: Optional[TensorLike]
    _spshape: Size

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
