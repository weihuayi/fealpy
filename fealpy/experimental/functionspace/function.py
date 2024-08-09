
from typing import Generic, Union

from ..backend import backend_manager as bm
from ..backend import TensorLike, Number
from .space import _FS, Index


class Function(Generic[_FS]):
    def __init__(self, space: _FS, array: TensorLike) -> None:
        self.space = space
        self.array = array

    def __call__(self, bcs: TensorLike, index: Index):
        return self.space.value(self.array, bcs, index)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.space}, {self.array})'

    def __getitem__(self, index: Index):
        return self.array[index]

    def __setitem__(self, index: Index, value: TensorLike):
        if bm.backend_name == 'jax':
            self.array.at[index].set(value)
        else:
            self.array[index] = value

    def __getattr__(self, item: str):
        if item in {'space', 'array'}:
            return object.__getattribute__(self, item)

        if item.endswith('value'):
            if hasattr(self.space, item):
                return getattr(self.space, item)

        return getattr(self.array, item)

    def __pos__(self):
        return self.__class__(self.space, +self.array)

    def __neg__(self):
        return self.__class__(self.space, -self.array)

    def __add__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, self.array + other)
    __radd__ = __add__

    def __iadd__(self, other: Union[TensorLike, Number]):
        self.array += other
        return self

    def __sub__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, self.array - other)

    def __rsub__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, other - self.array)

    def __isub__(self, other: Union[TensorLike, Number]):
        self.array -= other
        return self

    def __mul__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, self.array * other)
    __rmul__ = __mul__

    def __imul__(self, other: Union[TensorLike, Number]):
        self.array *= other
        return self

    def __truediv__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, self.array / other)

    def __rtruediv__(self, other: Union[TensorLike, Number]):
        return self.__class__(self.space, other / self.array)

    def __itruediv__(self, other: Union[TensorLike, Number]):
        self.array /= other
        return self

    def __matmul__(self, other: TensorLike) -> TensorLike:
        return self.array @ other

    def __rmatmul__(self, other: TensorLike) -> TensorLike:
        return other @ self.array
