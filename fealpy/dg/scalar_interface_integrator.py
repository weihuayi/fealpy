
import numpy as np

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalerInterfaceIntegrator():
    def __init__(self, q: int) -> None:
        self.q = q

    def assembly_face_vector(self, space: ScaledMonomialSpace, out=None):
        q = self.q
        mesh: PolygonMesh = space.mesh
        gdof = space.number_of_global_dofs()

        index = mesh.ds.boundary_face_flag()
        face2cell = mesh.ds.face_to_cell()[index, ...]
        in_face_flag = face2cell[:, 0] != face2cell[:, 1]
        fh = mesh.entity_measure('face')
        fn = mesh.entity_measure('face')
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.face_bc_to_point(bcs) #(NQ, NF, GD)

        phil = space.basis(ps, index=face2cell[:, 0]) # (NQ, NF, ldof)
        phir = space.basis(ps, index=face2cell[in_face_flag, 1]) # (NQ, in_NF, ldof)
        gphil = np.sum(
            space.grad_basis(ps, index=face2cell[:, 0]) * fn[None, :, None, :],
            axis=-1
        ) # (NQ, NF, ldof)
        gphir = np.sum(
            space.grad_basis(ps, index=face2cell[in_face_flag, 1]) * fn[None, :, None, :],
            axis=-1
        ) # (NQ, in_NF, ldof)
        Al = np.einsum('q, qfj, qfj, f -> fj', ws, phil, gphil, optimize=True) # (NF, ldof)
        Al += Al.T
        Ar = np.einsum('q, qfj, qfj, f -> fj', ws, phir, gphir, optimize=True) # (in_NF, ldof)
        Ar += Ar.T
        cell2dof = space.cell_to_dof()

        if out is None:
            F = np.zeros(gdof, dtype=mesh.ftype)
            np.add.at(F, cell2dof[face2cell[:, 0]], -Al)
            np.add.at(F, cell2dof[face2cell[in_face_flag, 1]], Ar)
            return F
        else:
            np.add.at(out, cell2dof[face2cell[:, 0]], -Al)
            np.add.at(out, cell2dof[face2cell[in_face_flag, 1]], Ar)
            return out
