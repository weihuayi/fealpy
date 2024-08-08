
from typing import Optional, Union, overload, Dict, Any, Self
from math import prod

from ..backend import TensorLike, Number, Size
from ..backend import backend_manager as bm
from .utils import _dense_ndim, _dense_shape


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

    def neg(self) -> Self:
        raise NotImplementedError

    @overload
    def add(self, other: Union[Number, Self], alpha: Number=1) -> Self: ...
    @overload
    def add(self, other: TensorLike, alpha: Number=1) -> TensorLike: ...
    def add(self, other: Union[Number, Self, TensorLike], alpha: Number=1) -> Union[Self, TensorLike]:
        raise NotImplementedError

    def mul(self, other: Union[Number, Self, TensorLike]) -> Self:
        raise NotImplementedError

    def div(self, other: Union[Number, TensorLike]) -> Self:
        raise NotImplementedError

    def pow(self, other: Union[TensorLike, Number]) -> Self:
        raise NotImplementedError

    @overload
    def matmul(self, other: Self) -> Self: ...
    @overload
    def matmul(self, other: TensorLike) -> TensorLike: ...
    def matmul(self, other: Union[Self, TensorLike]):
        raise NotImplementedError

    def __pos__(self): return self
    def __neg__(self): return self.neg()

    @overload
    def __add__(self, other: Union[Self, Number]) -> Self: ...
    @overload
    def __add__(self, other: TensorLike) -> TensorLike: ...
    def __add__(self, other):
        return self.add(other)

    @overload
    def __radd__(self, other: Union[Self, Number]) -> Self: ...
    @overload
    def __radd__(self, other: TensorLike) -> TensorLike: ...
    def __radd__(self, other):
        return self.add(other)

    @overload
    def __sub__(self, other: Union[Self, Number]) -> Self: ...
    @overload
    def __sub__(self, other: TensorLike) -> TensorLike: ...
    def __sub__(self, other):
        return self.add(-other)

    @overload
    def __rsub__(self, other: Union[Self, Number]) -> Self: ...
    @overload
    def __rsub__(self, other: TensorLike) -> TensorLike: ...
    def __rsub__(self, other):
        return self.neg().add(other)

    def __mul__(self, other: Union[Self, TensorLike, Number]) -> Self:
        return self.mul(other)
    __rmul__ = __mul__

    def __truediv__(self, other: Union[TensorLike, Number]) -> Self:
        return self.div(other)

    def __pow__(self, other: Union[TensorLike, Number]) -> Self:
        return self.pow(other)

    @overload
    def __matmul__(self, other: Self) -> Self: ...
    @overload
    def __matmul__(self, other: TensorLike) -> TensorLike: ...
    def __matmul__(self, other):
        return self.matmul(other)
