
from typing import Union, Callable

from torch import Tensor

from ..functionspace.space import FunctionSpace as _FunctionSpace

Index = Union[int, slice, Tensor]
_S = slice(None)


class Integrator():
    r"""@brief The base class for integrators on function spaces."""
    pass


class DomainIntegrator(Integrator):
    def assembly_cell_matrix(self, space: _FunctionSpace, index: Index=_S) -> Tensor:
        r"""Assembly matrix for each cell. Returns a tensor of shape (NC, ldof, ldof)."""
        raise NotImplementedError


class BoundaryIntegrator(Integrator):
    def assembly_face_matrix(self, space: _FunctionSpace, index: Index=_S) -> Tensor:
        r"""Assembly matrix for each face into global. Returns a tensor of shape (gdof, gdof)."""
        raise NotImplementedError


class DomainSourceIntegrator(Integrator):
    def assembly_cell_vector(self, space: _FunctionSpace, index: Index=_S) -> Tensor:
        raise NotImplementedError


class BoundarySourceIntegrator(Integrator):
    def assembly_face_vector(self, space: _FunctionSpace, index: Index=_S) -> Tensor:
        raise NotImplementedError


__all__ = [
    'Integrator',
    'DomainIntegrator',
    'BoundaryIntegrator',
    'DomainSourceIntegrator',
    'BoundarySourceIntegrator'
]

CoefLike = Union[float, int, Tensor, Callable[..., Tensor]]
