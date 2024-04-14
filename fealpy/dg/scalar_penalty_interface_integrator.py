
from typing import Optional
import numpy as np

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalerPenaltyInterfaceIntegrator():
    r"""Scalar boundary source integrator."""
    def __init__(self, q: int, gamma: Optional[float]=None) -> None:
        self.q = q
        self.gamma = gamma

    def assembly_face_vector(self, space: ScaledMonomialSpace, out=None):
        p = space.p
        q = self.q
        gamma = self.gamma if self.gamma is not None else p * (p+1)
        mesh: PolygonMesh = space.mesh
        gdof = space.number_of_global_dofs()

        index = mesh.ds.boundary_face_flag()
        face2cell = mesh.ds.face_to_cell()[index, ...]
        in_face_flag = face2cell[:, 0] != face2cell[:, 1]
        fh = mesh.entity_measure('face')
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.face_bc_to_point(bcs) #(NQ, bd_NF, GD)

        phil = space.basis(ps, index=face2cell[:, 0]) # (NQ, NF, ldof)
        phir = space.basis(ps, index=face2cell[in_face_flag, 1]) # (NQ, in_NF, ldof)
        Al = np.einsum('q, qfj, qfj, f -> fj', ws, phil, phil, 1/fh,  optimize=True) # (NQ, ldof)
        Ar = np.einsum('q, qfj, qfj, f -> fj', ws, phir, phir, 1/fh,  optimize=True)
        cell2dof = space.cell_to_dof()

        if out is None:
            F = np.zeros(gdof, dtype=mesh.ftype)
            np.add.at(F, cell2dof[face2cell[:, 0]], Al*gamma)
            np.add.at(F, cell2dof[face2cell[in_face_flag, 1]], Ar*gamma)
            return F
        else:
            np.add.at(out, cell2dof[face2cell[:, 0]], Al*gamma)
            np.add.at(out, cell2dof[face2cell[in_face_flag, 1]], Ar*gamma)
            return out
