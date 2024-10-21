
from typing import Optional
import numpy as np

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalarDirichletBoundarySourceIntegrator():
    r"""Scalar Dirichlet boundary source integrator."""
    def __init__(self, source, q: Optional[int]=None):
        self.source = source
        self.q = q

    def assembly_face_vector(self, space: ScaledMonomialSpace, out=None):
        q = self.q if self.q else space.p + 1
        mesh: PolygonMesh = space.mesh
        gdof = space.number_of_global_dofs()

        face2cell = mesh.ds.face_to_cell()
        bd_face_flag = face2cell[:, 0] == face2cell[:, 1]
        fn = mesh.face_unit_normal(index=bd_face_flag) # (bd_NF, GD)
        fm = mesh.entity_measure('face', index=bd_face_flag)
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.face_bc_to_point(bcs, index=bd_face_flag) # (NQ, bd_NF, GD)
        gval = self.source(ps) # (NQ, bd_NF)

        gphi = np.sum(
            space.grad_basis(ps, index=face2cell[bd_face_flag, 0])\
            * fn[None, :, None, :],
            axis=-1
        ) # (NQ, bd_NF, ldof)
        A = -np.einsum('q, qf, qfj, f -> fj', ws, gval, gphi, fm, optimize=True) # (NQ, ldof)
        cell2dof = space.cell_to_dof()

        if out is None:
            F = np.zeros((gdof, ), dtype=mesh.ftype)
            np.add.at(F, cell2dof[face2cell[bd_face_flag, 0]], A)
            return F
        else:
            np.add.at(out, cell2dof[face2cell[bd_face_flag, 0]], A)
            return out
