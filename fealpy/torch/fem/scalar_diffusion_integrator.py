
from typing import Optional

from torch import Tensor

from ..mesh import HomoMesh
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import DomainIntegrator, _FS, _S, Index, CoefLike


class ScalarDiffusionIntegrator(DomainIntegrator[_FS]):
    r"""The diffusion integrator for function spaces based on homogeneous meshes."""
    def __init__(self, c: Optional[CoefLike]=None, q: int=3, *,
                 batched: bool=False) -> None:
        self.coef = c
        self.q = q
        self.batched = batched

    def assembly_cell_matrix(self, space: _FS, index: Index=_S) -> Tensor:
        coef = self.coef
        q = self.q
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomoMesh):
            raise RuntimeError("The ScalarDiffusionIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index, variable='x')
        coef = process_coef_func(coef, mesh=mesh, index=index)

        return bilinear_integral(gphi, gphi, ws, cm, coef, batched=self.batched)
