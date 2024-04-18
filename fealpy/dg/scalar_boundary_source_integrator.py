
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalerBoundarySourceIntegrator():
    r"""Scalar boundary source integrator."""
    def __init__(self, source, c: Optional[float]=None, q: Optional[int]=None) -> None:
        self.source = source
        self.coef = c
        self.q = q

    def assembly_face_vector(self, space: ScaledMonomialSpace, out: Optional[NDArray]=None):
        coef = self.coef
        q = self.q
        mesh: PolygonMesh = space.mesh
        gdof = space.number_of_global_dofs()

        face2cell = mesh.ds.face_to_cell()
        bd_face_flag = face2cell[:, 0] == face2cell[:, 1]
        fm = mesh.entity_measure('face', index=bd_face_flag)
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.face_bc_to_point(bcs, index=bd_face_flag) #(NQ, bd_NF, GD)
        NQ, NF, GD = ps.shape
        gval = self.source(ps) #(NQ, bd_NF)
        phi = space.basis(ps, index=face2cell[bd_face_flag, 0]) # (NQ, bd_NF, ldof)

        if coef is None:
            A = np.einsum('q, qf, qfj, f -> fj', ws, gval, phi, fm, optimize=True) # (NQ, ldof)
        elif np.isscalar(coef):
            A = np.einsum('q, qf, qfj, f -> fj', ws, gval, phi, fm, optimize=True) * coef
        elif isinstance(coef, np.ndarray):
            if coef.shape == (NF, ):
                coef_subs = 'f'
            elif coef.shape == (NQ, NF):
                coef_subs = 'qf'
            else:
                raise ValueError(f'coef.shape = {coef.shape} is not supported.')
            A = np.einsum(f'q, qf, qfj, {coef_subs}, f -> fj', ws, gval, phi, coef, fm, optimize=True)
        cell2dof = space.cell_to_dof()

        if out is None:
            F = np.zeros((gdof, ), dtype=mesh.ftype)
            np.add.at(F, cell2dof[face2cell[bd_face_flag, 0]], A)
            return F
        else:
            np.add.at(out, cell2dof[face2cell[bd_face_flag, 0]], A)
            return out
