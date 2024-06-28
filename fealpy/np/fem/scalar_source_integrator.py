
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import linear_integral
from .integrator import CellSourceIntegrator, _S, Index, CoefLike, enable_cache


class ScalarSourceIntegrator(CellSourceIntegrator):
    r"""The domain source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, source: Optional[CoefLike]=None, q: int=3, *,
                 index: Index=_S) -> None:
        super().__init__()
        self.source = source
        self.q = q
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> NDArray:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarSourceIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index, variable='x')

        return bcs, ws, phi, cm, index

    def assembly(self, space: _FS) -> NDArray:
        f = self.source
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
        val = process_coef_func(f, bcs=bcs, mesh=mesh, etype='cell', index=index)

        return linear_integral(phi, ws, cm, val)
