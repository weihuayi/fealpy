from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import FaceOperatorIntegrator, _S, Index, enable_cache


class ScalarRobinBoundaryIntegrator(FaceOperatorIntegrator):
    def __init__(self, kappa, q=3, *, 
                 threshold: Optional[Callable[[NDArray], NDArray]]=None):
        super().__init__()
        self.kappa = kappa
        self.q = q

    @enable_cache
    def to_global_dof(self, space: _FS) -> NDArray:
        index = space.mesh.boundary_face_index()
        return space.face_to_dof()[index]

    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarRobinBoundaryIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        index = mesh.boundary_face_index()
        fm = mesh.entity_measure('face', index=index)
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index, variable='x')

        return bcs, ws, phi, fm, index

    def assembly(self, space: _FS):

        kappa = self.kappa
        mesh = space.mesh
        bcs, ws, phi, fm, index = self.fetch(space)
        val = process_coef_func(kappa, bcs=bcs, mesh=mesh, etype='cell', index=index)
        return bilinear_integral(phi, phi, ws, fm, val)