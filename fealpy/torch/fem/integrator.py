
from typing import Union, Generic, TypeVar, Callable

from torch import Tensor

from ..functionspace.space import FunctionSpace

_FS = TypeVar('_FS', bound=FunctionSpace)
Index = Union[int, slice, Tensor]
_S = slice(None)


class Integrator(Generic[_FS]):
    r"""@brief The base class for integrators on function spaces."""
    def assembly(self, space: _FS, index: Index=_S) -> Tensor:
        r"""Assembly the matrix or vector."""
        raise NotImplementedError


class DomainIntegrator(Integrator[_FS]):
    def assembly_cell_matrix(self, space: _FS, index: Index=_S) -> Tensor:
        r"""Assembly matrix for each cell. Returns a tensor of shape (NC, ldof, ldof)."""
        raise NotImplementedError


class BoundaryIntegrator(Integrator[_FS]):
    def assembly_face_matrix(self, space: _FS, index: Index=_S) -> Tensor:
        r"""Assembly matrix for each face into global. Returns a tensor of shape (gdof, gdof)."""
        raise NotImplementedError


class DomainSourceIntegrator(Integrator[_FS]):
    def assembly_cell_vector(self, space: _FS, index: Index=_S) -> Tensor:
        raise NotImplementedError


class BoundarySourceIntegrator(Integrator[_FS]):
    def assembly_face_vector(self, space: _FS, index: Index=_S) -> Tensor:
        raise NotImplementedError


__all__ = [
    'Integrator',
    'DomainIntegrator',
    'BoundaryIntegrator',
    'DomainSourceIntegrator',
    'BoundarySourceIntegrator'
]

CoefLike = Union[float, int, Tensor, Callable[..., Tensor]]
