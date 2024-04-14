
import numpy as np

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalarDirichletBoundarySourceIntegrator():
    def __init__(self, threshold=None, q=3):
        self.threshold = threshold
        self.q = q

    def assembly_face_vector(self, space: ScaledMonomialSpace, out=None):
        q = self.q
        mesh: PolygonMesh = space.mesh
        gdof = space.number_of_global_dofs()

        index = mesh.ds.boundary_face_flag()
        face2cell = mesh.ds.face_to_cell()[index, ...]
        bd_face_flag = face2cell[:, 0] == face2cell[:, 1]
        fn = mesh.face_unit_normal(index=bd_face_flag) # (bd_NF, GD)
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.face_bc_to_point(bcs, index=bd_face_flag) # (NQ, bd_NF, GD)
        gval = self.source(ps) # (NQ, bd_NF)

        gphi = -np.sum(
            space.grad_basis(ps, index=face2cell[bd_face_flag, 0])\
            * fn[None, :, None, :],
            axis=-1
        ) # (NQ, bd_NF, ldof)
        A = np.einsum('q, qf, qfj -> fj', ws, gval, gphi, optimize=True) # (NQ, ldof)
        cell2dof = space.cell_to_dof()

        if out is None:
            F = np.zeros(gdof, dtype=mesh.ftype)
            np.add.at(F, cell2dof[face2cell[bd_face_flag, 0]], A)
            return F
        else:
            np.add.at(out, cell2dof[face2cell[bd_face_flag, 0]], A)
            return out
