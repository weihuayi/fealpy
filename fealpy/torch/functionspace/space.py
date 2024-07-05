
from typing import Union, Callable, Optional, Generic, TypeVar

from ..typing import Tensor, _dtype, _device, Index, Number, _S, Size
from .utils import zero_dofs


class _FunctionSpace():
    r"""The base class of function spaces"""
    device: _device
    ftype: _dtype
    itype: _dtype

    # basis
    def basis(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor: raise NotImplementedError
    def grad_basis(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor: raise NotImplementedError
    def hess_basis(self, p: Tensor, index: Index=_S, **kwargs) -> Tensor: raise NotImplementedError

    # values
    def value(self, uh: Tensor, p: Tensor, index: Index=_S) -> Tensor: raise NotImplementedError
    def grad_value(self, uh: Tensor, p: Tensor, index: Index=_S) -> Tensor: raise NotImplementedError

    # counters
    def number_of_global_dofs(self) -> int: raise NotImplementedError
    def number_of_local_dofs(self, doftype='cell') -> int: raise NotImplementedError

    # relationships
    def cell_to_dof(self) -> Tensor: raise NotImplementedError
    def face_to_dof(self) -> Tensor: raise NotImplementedError

    # interpolation
    def interpolate(self, source: Union[Callable[..., Tensor], Tensor, Number],
                    uh: Tensor, dim: Optional[int]=None, index: Index=_S) -> Tensor:
        raise NotImplementedError

    # function
    def array(self, dim: Union[Size, int, None]=None) -> Tensor:
        """Initialize a Tensor filled with zeros as values of DoFs.

        Parameters:
            dim (Tuple[int, ...] | int | None, optional): Shape of DoFs. Defaults to None.

        Returns:
            Tensor: Values of DoFs shaped (GDOF, *dim).
        """
        GDOF = self.number_of_global_dofs()
        return zero_dofs(GDOF, dim, dtype=self.ftype, device=self.device)


_FS = TypeVar('_FS', bound=_FunctionSpace)


class Function(Tensor, Generic[_FS]):
    space: _FS

    # NOTE: Named tensors and all their associated APIs are an experimental feature
    # and subject to change. Please do not use them for anything important until
    # they are released as stable.
    @staticmethod
    def __new__(cls, space: _FS, tensor: Tensor) -> Tensor:
        assert isinstance(space, _FunctionSpace)
        tensor = tensor.to(device=space.device, dtype=space.ftype)
        return Tensor._make_subclass(cls, tensor)

    def __init__(self, space: _FS, tensor: Tensor) -> None:
        self.space = space

    def __call__(self, bc: Tensor, index=_S) -> Tensor:
        return self.space.value(self, bc, index)

    # NOTE: Some methods and attributes of Tensor are very similar to those of FunctionSpace.
    # Such as `values()`, `grad`.

    def grad_value(self, bc: Tensor, index=_S):
        return self.space.grad_value(self, bc, index)

    def interpolate_(self, source: Union[Callable[..., Tensor], Tensor, Number],
                     dim: Optional[int]=None, index: Index=_S) -> Tensor:
        return self.space.interpolate(source, self, dim, index)


class FunctionSpace(_FunctionSpace):
    def function(self, tensor: Optional[Tensor]=None, dim: Union[Size, int, None]=None) -> Tensor:
        if tensor is None:
            tensor = self.array(dim=dim)

        func_ = Function(self, tensor)

        return func_
