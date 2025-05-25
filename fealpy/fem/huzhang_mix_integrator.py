
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

class HuZhangMixIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, q = None): 
        super().__init__()
        self.q = q

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        c2d1  = space[1].cell_to_dof()
        c2d0  = space[0].cell_to_dof()
        return (c2d0, c2d1)

    @enable_cache
    def fetch(self, space: _FS):
        space0 = space[0]
        space1 = space[1]

        p    = space0.p
        mesh = space0.mesh
        TD = mesh.top_dimension()
        qf = mesh.quadrature_formula(p+3, 'cell')
        cm = mesh.entity_measure('cell')

        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space1.div_basis(bcs)
        psi = space0.basis(bcs)
        return cm, phi, psi, ws

    def assembly(self, space: _FS) -> TensorLike:
        assert space[0].mesh == space[1].mesh, "The mesh should be same for two space "

        mesh = getattr(space[0], 'mesh', None)
        cm, phi, psi, ws = self.fetch(space)
        res = bm.einsum('q, c, cqld, cqmd->clm', ws, cm, phi, psi)
        #res = res.reshape(res.shape[:-2]+(-1, ))
        #res = bm.einsum("clm->cml", res)
        return res



