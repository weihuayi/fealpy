
from typing import Union, Callable, Optional, Any

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, Number, _S, Size
from .function import Function
from .utils import zero_dofs


class FunctionSpace():
    r"""The base class of function spaces"""
    ftype: Any
    itype: Any
    device: Any

    # basis
    def basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike: raise NotImplementedError
    def grad_basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike: raise NotImplementedError
    def hess_basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike: raise NotImplementedError

    # values
    def value(self, uh: TensorLike, p: TensorLike, index: Index=_S) -> TensorLike: raise NotImplementedError
    def grad_value(self, uh: TensorLike, p: TensorLike, index: Index=_S) -> TensorLike: raise NotImplementedError

    # counters
    def number_of_global_dofs(self) -> int: raise NotImplementedError
    def number_of_local_dofs(self, doftype='cell') -> int: raise NotImplementedError

    # relationships
    def cell_to_dof(self, index: Index=_S) -> TensorLike: raise NotImplementedError
    def face_to_dof(self, index: Index=_S) -> TensorLike: raise NotImplementedError

    # interpolation
    def interpolate(self, source: Union[Callable[..., TensorLike], TensorLike, Number],
                    uh: TensorLike, dim: Optional[int]=None, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    def array(self, batch: Union[int, Size, None]=None, *, dtype=None, device=None) -> TensorLike:
        """Initialize a Tensor filled with zeros as values of DoFs.

        Parameters:
            batch (int | Size | None, optional): shape of the batch.

        Returns:
            Tensor: Values of DoFs shaped (batch, GDOF).
        TODO:
            1. device
        """
        GDOF = self.number_of_global_dofs()
        if (batch is None) or (batch == 0):
            batch = tuple()

        elif isinstance(batch, int):
            batch = (batch, )

        shape = batch + (GDOF, )

        if dtype is None:
            dtype = self.ftype 

        return bm.zeros(shape, dtype=dtype, device=device)

    def function(self, array: Optional[TensorLike]=None,
                batch: Union[int, Size, None]=None, *,
                coordtype='barycentric', 
                dtype=None, device=None):
        """Initialize a Function in the space.

        Parameters:

        Returns:
            Function: A Function object.
        """
        if array is None:
            if dtype is None:
                dtype = self.ftype
            if device is None:
                device = self.device
            array = self.array(batch=batch, dtype=dtype, device=device)

        return Function(self, array, coordtype)
