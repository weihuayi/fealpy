
from typing import Generic, TypeVar, Optional, List, overload, Sequence

import torch
from torch import Tensor

from .. import logger
from ..functionspace.space import FunctionSpace
from .form import IntegratorHandler
from .integrator import DomainIntegrator as _DI
from .integrator import BoundaryIntegrator as _BI


_FS = TypeVar('_FS', bound=FunctionSpace)


class BilinearForm(Generic[_FS]):
    r"""@brief"""
    def __init__(self, space: _FS, retain_ints: bool=False, batched: bool=False):
        self.space = space
        self.dintegrators: List[IntegratorHandler] = []
        self.bintegrators: List[IntegratorHandler] = []
        self._M: Optional[Tensor] = None
        self.retain_ints = retain_ints
        self.batched = batched

    def __len__(self) -> int:
        return len(self.dintegrators) + len(self.bintegrators)

    @overload
    def add_domain_integrator(self, I: _DI[_FS]) -> IntegratorHandler: ...
    @overload
    def add_domain_integrator(self, I: Sequence[_DI[_FS]]) -> List[IntegratorHandler]: ...
    @overload
    def add_domain_integrator(self, *I: _DI[_FS]) -> List[IntegratorHandler]: ...
    def add_domain_integrator(self, *I):
        if len(I) == 1:
            I = I[0]
            if isinstance(I, Sequence):
                handler = IntegratorHandler.build_list(I, form=self)
                self.dintegrators.extend(handler)
            else:
                handler = IntegratorHandler(I, form=self)
                self.dintegrators.append(handler)
        elif len(I) >= 2:
            handler = IntegratorHandler.build_list(I, form=self)
            self.dintegrators.extend(handler)
        else:
            raise RuntimeError("add_domain_integrator() is called with no arguments.")

        return handler

    @overload
    def add_boundary_integrator(self, I: _BI[_FS]) -> IntegratorHandler: ...
    @overload
    def add_boundary_integrator(self, I: Sequence[_BI[_FS]]) -> List[IntegratorHandler]: ...
    @overload
    def add_boundary_integrator(self, *I: _BI[_FS]) -> List[IntegratorHandler]: ...
    def add_boundary_integrator(self, *I):
        if len(I) == 1:
            I = I[0]
            if isinstance(I, Sequence):
                handler = IntegratorHandler.build_list(I, form=self)
                self.bintegrators.extend(handler)
            else:
                handler = IntegratorHandler(I, form=self)
                self.bintegrators.append(handler)
        elif len(I) >= 2:
            handler = IntegratorHandler.build_list(I, form=self)
            self.bintegrators.extend(handler)
        else:
            raise RuntimeError("add_boundary_integrator() is called with no arguments.")

        return handler

    def number_of_domain_integrators(self) -> int:
        return len(self.dintegrators)

    def number_of_boundary_integrators(self) -> int:
        return len(self.bintegrators)

    def assembly(self) -> Tensor:
        r"""Assembly the bilinear form matrix. Returns COO Tensor of shape (gdof, gdof)."""
        space = self.space
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()
        cell2dof = space.cell_to_dof()
        cm = torch.zeros((gdof, gdof), dtype=space.dtype, device=space.device)
        CM = self.dintegrators[0].assembly_cell_matrix(space)

        for idx, di in enumerate(self.dints):
            if di is None:
                self.render('d', idx)


        I = torch.broadcast_to(cell2dof[:, :, None], size=CM.shape)
        J = torch.broadcast_to(cell2dof[:, None, :], size=CM.shape)
        indices = torch.stack([I.ravel(), J.ravel()], dim=0)
        M = torch.sparse_coo_tensor(indices, CM.ravel(), size=(gdof, gdof))

        for bi in self.bintegrators:
            M = M + bi.assembly_face_matrix(space)

        self._M = M.coalesce()
        logger.info(f"Bilinear form matrix constructed, with shape {list(self._M.shape)}.")

        return self._M
