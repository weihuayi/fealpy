from ..backend import backend_manager as bm
from typing import Optional

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import LinearInt, OpInt, CellInt, enable_cache
from ..typing import TensorLike, Index, _S
from ..functionspace.functional import symmetry_span_array, symmetry_index

from ..mesh import TriangleMesh

from sympy import symbols, sin, cos, Matrix, lambdify

class HuZhangStressIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, q = None, lambda0 = 1.0, lambda1 = 1.0):
        super().__init__()
        self.q = q
        self.lambda0 = lambda0
        self.lambda1 = lambda1

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        c2d0  = space.cell_to_dof()
        return c2d0

    @enable_cache
    def fetch(self, space: _FS):
        p = space.p
        q = self.q if self.q else p+3

        mesh = space.mesh
        TD = mesh.top_dimension()
        cm = mesh.entity_measure('cell')
        qf = mesh.quadrature_formula(q, 'cell')

        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs)
        if TD == 2:
            trphi = phi[..., 0] + phi[..., -1]
        if TD == 3:
            trphi = phi[..., 0] + phi[..., 3] + phi[..., -1]
        return cm, phi, trphi, ws 

    def assembly(self, space: _FS) -> TensorLike:
        mesh = space.mesh 
        TD = mesh.top_dimension()
        lambda0, lambda1 = self.lambda0, self.lambda1 
        cm, phi, trphi, ws = self.fetch(space) 

        _, num = symmetry_index(d=TD, r=2)
        A  = lambda0*bm.einsum('q, c, cqld, cqmd, d->clm', ws, cm, phi, phi, num)
        A -= lambda1*bm.einsum('q, c, cql, cqm->clm', ws, cm, trphi, trphi)
        return A




