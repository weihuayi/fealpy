
from typing import Union, Callable, Optional

from torch import Tensor

from ..functionspace.space import FunctionSpace as _FunctionSpace

Index = Union[int, slice, Tensor]
_S = slice(None)


class Integrator():
    r"""@brief The base class for integrators on function spaces."""
    _value: Optional[Tensor]
    _assembly: str

    def assembly(self, space: _FunctionSpace) -> Tensor:
        if hasattr(self, '_value') and self._value is not None:
            return self._value
        else:
            if not hasattr(self, '_assembly'):
                raise NotImplementedError("Assembly method not defined.")
            if self._assembly == 'assembly':
                raise ValueError("Can not use assembly method name 'assembly'.")
            self._value = getattr(self, self._assembly)(space)
            return self._value

    def clear(self):
        self._value = None


class DomainIntegrator(Integrator):
    _assembly = 'assembly_cell_matrix'
    def assembly_cell_matrix(self, space: _FunctionSpace, index: Index=_S) -> Tensor:
        r"""Assembly matrix for each cell. Returns a tensor of shape (NC, ldof, ldof)."""
        raise NotImplementedError


class BoundaryIntegrator(Integrator):
    _assembly = 'assembly_face_matrix'
    def assembly_face_matrix(self, space: _FunctionSpace, index: Index=_S) -> Tensor:
        r"""Assembly matrix for each face into global. Returns a tensor of shape (gdof, gdof)."""
        raise NotImplementedError


class DomainSourceIntegrator(Integrator):
    _assembly = 'assembly_cell_vector'
    def assembly_cell_vector(self, space: _FunctionSpace, index: Index=_S) -> Tensor:
        raise NotImplementedError


class BoundarySourceIntegrator(Integrator):
    _assembly = 'assembly_face_vector'
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
