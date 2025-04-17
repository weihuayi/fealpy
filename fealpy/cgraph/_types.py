from typing import (
    Tuple, Literal, Union, Callable,
    Protocol
)

from ..backend import TensorLike as _DT


class SupportsAddNode(Protocol):
    def add_node(node) -> None: ...

TensorOrFunc = Union[_DT, Callable[[_DT], _DT]]


class PDEData(Protocol):
    solution        : TensorOrFunc
    gradient        : TensorOrFunc
    source          : TensorOrFunc
    dirichlet       : TensorOrFunc
    is_dirichlet_bc : TensorOrFunc
    neumann         : TensorOrFunc
    is_neumann_bc   : TensorOrFunc
    robin           : TensorOrFunc
    is_robin_bc     : TensorOrFunc

class SupportsIntegral(Protocol):
    def integral(self, f: TensorOrFunc, *, q=3, **kwargs) -> _DT: ...

class SupportsError(Protocol):
    def error(self, u:TensorOrFunc, v:TensorOrFunc, *, q=3, power=2, **kwargs) -> _DT: ...
