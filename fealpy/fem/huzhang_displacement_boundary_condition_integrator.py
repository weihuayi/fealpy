
from typing import Optional, Literal

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, SourceLike

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import linear_integral
from .integrator import LinearInt, SrcInt, CellInt, enable_cache, assemblymethod

class ScalarSourceIntegrator(LinearInt, SrcInt, CellInt):
    r"""The domain source integrator for function spaces based on homogeneous meshes."""
    def __init__(self, g: Optional[SourceLike]=None, q: int=None) -> None:
        super().__init__()
        self.g = g
        self.q = q 

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        bdface = mesh.boundary_face_index()
        f2c = mesh.face_to_cell()[bdface]
        cell2dof = space.cell_to_dof()[f2c[:, 0]]
        return cell2dof 

    @enable_cache
    def fetch(self, space: _FS, /, inidces=None):
        pass

    def assembly(self, space: _FS, indices=None) -> TensorLike:
        '''
        Notes
        -----
        Only support 3D case.
        '''
        g = self.source
        p = space.p
        q = self.q if self.q is not None else p+3

        mesh = space.mesh
        TD = mesh.top_dimension()
        ldof = space.number_of_local_dofs()
        gdof = space.number_of_global_dofs()

        bdface = mesh.boundary_face_index()
        f2c = mesh.face_to_cell()[bdface]
        fn  = mesh.face_unit_normal()[bdface]
        cell2dof = space.cell_to_dof()[f2c[:, 0]]
        NBF = len(bdface)

        cellmeasure = mesh.entity_measure('face')[bdface]
        qf = mesh.quadrature_formula(p+3, 'face')

        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(bcs)

        bcsi = [bm.insert(bcs, i, 0, axis=-1) for i in range(4)]

        symidx = [[0, 1, 2], [1, 3, 4], [2, 4, 5]]
        phin = bm.zeros((NBF, NQ, ldof, 3), dtype=space.ftype)
        gval = bm.zeros((NBF, NQ, 3), dtype=space.ftype)
        for i in range(4):
            flag = f2c[:, 2] == i
            phi = space.basis(bcsi[i])[f2c[flag, 0]] 
            phin[flag, ..., 0] = bm.sum(phi[..., symidx[0]] * fn[flag, None, None], axis=-1)
            phin[flag, ..., 1] = bm.sum(phi[..., symidx[1]] * fn[flag, None, None], axis=-1)
            phin[flag, ..., 2] = bm.sum(phi[..., symidx[2]] * fn[flag, None, None], axis=-1)
            points = mesh.bc_to_point(bcsi[i])[f2c[flag, 0]]
            gval[flag] = g(points)

        b = bm.einsum('q, c, cqld, cqd->cl', ws, cellmeasure, phin, gval)
        return b 


