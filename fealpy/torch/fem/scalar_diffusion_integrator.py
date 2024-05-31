
from typing import Optional

from torch import Tensor

from ..mesh import HomoMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import CellOperatorIntegrator, _S, Index, CoefLike


class ScalarDiffusionIntegrator(CellOperatorIntegrator):
    r"""The diffusion integrator for function spaces based on homogeneous meshes."""
    def __init__(self, c: Optional[CoefLike]=None, q: int=3, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.coef = c
        self.q = q
        self.index = index
        self.batched = batched

    def to_global_dof(self, space: _FS) -> Tensor:
        return space.cell_to_dof()[self.index]

    def assembly(self, space: _FS) -> Tensor:
        coef = self.coef
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomoMesh):
            raise RuntimeError("The ScalarDiffusionIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index, variable='x')
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return bilinear_integral(gphi, gphi, ws, cm, coef, batched=self.batched)
