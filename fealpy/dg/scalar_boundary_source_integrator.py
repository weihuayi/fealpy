
from typing import Optional
import numpy as np

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalerBoundarySourceIntegrator():
    r"""Scalar boundary source integrator."""
    def __init__(self, source, q: int, gamma: Optional[float]=None) -> None:
        self.source = source
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
        bd_face_flag = face2cell[:, 0] == face2cell[:, 1]
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.face_bc_to_point(bcs, index=bd_face_flag) #(NQ, bd_NF, GD)
        gval = self.source(ps) #(NQ, bd_NF)

        phi = space.basis(ps, index=face2cell[bd_face_flag, 0]) # (NQ, bd_NF, ldof)
        A = np.einsum('q, qf, qfj -> fj', ws, gval, phi,  optimize=True) # (NQ, ldof)
        cell2dof = space.cell_to_dof()

        if out is None:
            F = np.zeros(gdof, dtype=mesh.ftype)
            np.add.at(F, cell2dof[face2cell[bd_face_flag, 0]], A*gamma)
            return F
        else:
            np.add.at(out, cell2dof[face2cell[bd_face_flag, 0]], A*gamma)
            return out
