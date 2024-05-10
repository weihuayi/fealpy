
from typing import Optional

import torch
from torch import Tensor

from ..mesh import HomoMesh
from ..utils import process_coef_func, is_scalar
from .integrator import DomainIntegrator, _FS, _S, Index, CoefLike


class ScalarDiffusionIntegrator(DomainIntegrator[_FS]):
    r"""The diffusion integrator for function spaces based on homogeneous meshes."""
    def __init__(self, c: Optional[CoefLike]=None, q: int=3) -> None:
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, space: _FS, index: Index=_S) -> Tensor:
        coef = self.coef
        q = self.q
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomoMesh):
            raise RuntimeError("The ScalarDiffusionIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index)
        NC = cm.size(0)
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = ws.size(0)

        gphi = space.grad_basis(bcs, index, variable='x')

        if coef is None:
            return torch.einsum('q, qci..., qcj..., c -> cij', ws, gphi, gphi, cm)
        else:
            coef = process_coef_func(coef, mesh=mesh, index=index)
            if is_scalar(coef):
                return torch.einsum('q, qci..., qcj..., c -> cij', ws, gphi, gphi, cm) * coef
            else:
                if coef.shape == (NC, ):
                    return torch.einsum('q, qci..., qcj..., c -> cij', ws, gphi, gphi, cm*coef)
                elif coef.shape == (NQ, NC):
                    return torch.einsum('q, qci..., qcj..., c, qc -> cij', ws, gphi, gphi, cm, coef)
                else:
                    RuntimeError(f'coef shape {coef.shape} is not supported.')
