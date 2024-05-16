
from typing import Generic, TypeVar, Optional, List, overload, Sequence

import torch
from torch import Tensor

from .. import logger
from ..functionspace.space import FunctionSpace
from .integrator import DomainIntegrator as _DI
from .integrator import BoundaryIntegrator as _BI


_FS = TypeVar('_FS', bound=FunctionSpace)


class BilinearForm(Generic[_FS]):
    r"""@brief"""
    def __init__(self, space: _FS):
        self.space = space
        self.dintegrators: List[_DI[_FS]] = []
        self.bintegrators: List[_BI[_FS]] = []
        self._M: Optional[Tensor] = None

    @overload
    def add_domain_integrator(self, I: _DI[_FS]) -> None: ...
    @overload
    def add_domain_integrator(self, I: Sequence[_DI[_FS]]) -> None: ...
    @overload
    def add_domain_integrator(self, *I: _DI[_FS]) -> None: ...
    def add_domain_integrator(self, *I):
        if len(I) == 1:
            I = I[0]
            if isinstance(I, Sequence):
                self.dintegrators.extend(I)
            else:
                self.dintegrators.append(I)
        elif len(I) >= 2:
            self.dintegrators.extend(I)
        else:
            logger.warning("add_domain_integrator() is called with no arguments.")

    @overload
    def add_boundary_integrator(self, I: _BI[_FS]) -> None: ...
    @overload
    def add_boundary_integrator(self, I: Sequence[_BI[_FS]]) -> None: ...
    @overload
    def add_boundary_integrator(self, *I: _BI[_FS]) -> None: ...
    def add_boundary_integrator(self, *I):
        if len(I) == 1:
            I = I[0]
            if isinstance(I, Sequence):
                self.bintegrators.extend(I)
            else:
                self.bintegrators.append(I)
        elif len(I) >= 2:
            self.bintegrators.extend(I)
        else:
            logger.warning("add_boundary_integrator() is called with no arguments.")

    def assembly(self) -> Tensor:
        r"""Assembly the bilinear form matrix. Returns COO Tensor of shape (gdof, gdof)."""
        space = self.space
        gdof = space.number_of_global_dofs()
        CM = self.dintegrators[0].assembly_cell_matrix(space)

        for di in self.dintegrators[1:]:
            CM = CM + di.assembly_cell_matrix(space)

        cell2dof = space.cell_to_dof()
        I = torch.broadcast_to(cell2dof[:, :, None], size=CM.shape)
        J = torch.broadcast_to(cell2dof[:, None, :], size=CM.shape)
        indices = torch.stack([I.ravel(), J.ravel()], dim=0)
        M = torch.sparse_coo_tensor(indices, CM.ravel(), size=(gdof, gdof))

        for bi in self.bintegrators:
            M = M + bi.assembly_face_matrix(space)

        self._M = M.coalesce()
        logger.info(f"Bilinear form matrix constructed, with shape {list(self._M.shape)}.")

        return self._M
