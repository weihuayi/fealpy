
from typing import Optional

from torch import Tensor

from ..mesh import HomoMesh
from ..utils import process_coef_func
from ..functional import linear_integral
from .integrator import DomainSourceIntegrator, _FS, _S, Index, CoefLike


class ScalarSourceIntegrator(DomainSourceIntegrator):
    r"""The domain source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, source: Optional[CoefLike]=None, q: int=3, *,
                 batched: bool=False):
        self.f = source
        self.q = q
        self.batched = batched

    def assembly_cell_vector(self, space: _FS, index: Index=_S) -> Tensor:
        f = self.f
        q = self.q
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
