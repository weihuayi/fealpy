
from typing import Callable, Union

import torch
from torch import Tensor

from ..mesh import HomoMesh
from ..utils import process_coef_func, is_scalar
from .integrator import DomainSourceIntegrator, _FS, _S, Index


class ScalarSourceIntegrator(DomainSourceIntegrator):
    def __init__(self, source: Union[Callable, int, float, Tensor], q: int=3):
        r""""""
        self.f = source
        self.q = q

    def assembly_cell_vector(self, space: _FS, index: Index=_S):
        f = self.f
        q = self.q
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomoMesh):
            raise RuntimeError("The ScalarSourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        NC = cm.size(0)
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = ws.size(0)

        phi = space.basis(bcs, index=index, variable='x')
        val = process_coef_func(f, bcs, mesh, index)

        if is_scalar(val):
            return val * torch.einsum('q, qci, c -> ci', ws, phi, cm)
        else:
            if val.shape == (NC, ):
                return torch.einsum('q, c, qci, c -> ci', ws, val, phi, cm)
            elif val.shape == (NQ, NC):
                return torch.einsum('q, qc, qci, c -> ci', ws, val, phi, cm)
            else:
                raise RuntimeError(f'source value shape {val.shape} is not supported.')
