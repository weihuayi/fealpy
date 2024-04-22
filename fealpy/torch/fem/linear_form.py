
from typing import Generic, TypeVar, Optional, List, overload, Sequence

import torch
from torch import Tensor

from .. import logger
from ..functionspace.space import FunctionSpace
from .integrator import DomainSourceIntegrator as _DSI
from .integrator import BoundarySourceIntegrator as _BSI


_FS = TypeVar('_FS', bound=FunctionSpace)


class LinearForm(Generic[_FS]):
    r"""@brief"""
    def __init__(self, space: _FS):
        self.space = space
        self.dintegrators: List[_DSI[_FS]] = []
        self.bintegrators: List[_BSI[_FS]] = []
        self._M: Optional[Tensor] = None

    @overload
    def add_domain_integrator(self, I: _DSI[_FS]) -> None: ...
    @overload
    def add_domain_integrator(self, I: Sequence[_DSI[_FS]]) -> None: ...
    @overload
    def add_domain_integrator(self, *I: _DSI[_FS]) -> None: ...
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
    def add_boundary_integrator(self, I: _BSI[_FS]) -> None: ...
    @overload
    def add_boundary_integrator(self, I: Sequence[_BSI[_FS]]) -> None: ...
    @overload
    def add_boundary_integrator(self, *I: _BSI[_FS]) -> None: ...
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
        r"""Assembly the linear form vector. Returns COO Tensor of shape (gdof,)."""
        space = self.space
        cell2dof = space.cell_to_dof()
        NC = cell2dof.shape[0]
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()

        bb = torch.zeros((NC, ldof), dtype=space.ftype)
        bb = self.dintegrators[0].assembly_cell_vector(space)

        for di in self.dintegrators[1:]:
            bb = bb + di.assembly_cell_vector(space)

        indices = cell2dof.ravel().unsqueeze(0)
        V = torch.sparse_coo_tensor(indices, bb.ravel(), (gdof,))

        for bi in self.bintegrators:
            V = V + bi.assembly_face_vector(space)

        V.coalesce()
        self._V = V
        logger.info(f"Construct source vector with shape {V.shape}.")
        return V
