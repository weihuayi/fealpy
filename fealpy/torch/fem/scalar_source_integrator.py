
from typing import Optional

from torch import Tensor

from ..mesh import HomoMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import linear_integral
from .integrator import CellSourceIntegrator, _S, Index, CoefLike


class ScalarSourceIntegrator(CellSourceIntegrator):
    r"""The domain source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, source: Optional[CoefLike]=None, q: int=3, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__(index=index)
        self.f = source
        self.q = q
        self.batched = batched

    def to_global_dof(self, space: _FS) -> Tensor:
        return space.cell_to_dof()[self.index]

    def assembly(self, space: _FS) -> Tensor:
        f = self.f
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomoMesh):
            raise RuntimeError("The ScalarSourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index, variable='x')
        val = process_coef_func(f, bcs, mesh, index)

        return linear_integral(phi, ws, cm, val, batched=self.batched)
