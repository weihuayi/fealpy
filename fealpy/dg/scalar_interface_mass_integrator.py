
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalerInterfaceMassIntegrator():
    r"""Scalar boundary source integrator."""
    def __init__(self, q: int, coef: Union[NDArray, float, None]=None) -> None:
        self.q = q
        self.coef = coef

    def assembly_face_vector(self, space: ScaledMonomialSpace, out=None):
        q = self.q
        coef = self.coef
        mesh: PolygonMesh = space.mesh
        gdof = space.number_of_global_dofs()

        index = mesh.ds.boundary_face_flag()
        face2cell = mesh.ds.face_to_cell()[index, ...]
        in_face_flag = face2cell[:, 0] != face2cell[:, 1]
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.face_bc_to_point(bcs) #(NQ, bd_NF, GD)
        NQ, NF, GD = ps.shape

        phil = space.basis(ps, index=face2cell[:, 0]) # (NQ, NF, ldof)
        phir = space.basis(ps, index=face2cell[in_face_flag, 1]) # (NQ, in_NF, ldof)

        if coef is None:
            Al = np.einsum('q, qfj, qfj -> fj', ws, phil, phil, optimize=True) # (NQ, ldof)
            Ar = np.einsum('q, qfj, qfj -> fj', ws, phir, phir, optimize=True)
        elif np.isscalar(coef):
            Al = np.einsum('q, qfj, qfj -> fj', ws, phil, phil, optimize=True) * coef # (NQ, ldof)
            Ar = np.einsum('q, qfj, qfj -> fj', ws, phir, phir, optimize=True) * coef
        elif isinstance(coef, np.ndarray):
            if coef.shape == (NF, ):
                coef_subs = 'c'
            elif coef.shape == (NQ, NF):
                coef_subs = 'qc'
            elif coef.shape == (GD, GD):
                coef_subs = 'ij'
            else:
                raise ValueError(f'coef.shape = {coef.shape} is not supported.')
            Al = np.einsum(f'q, {coef_subs}, qfj, qfj -> fj', ws, coef, phil, phil, optimize=True) # (NQ, ldof)
            Ar = np.einsum(f'q, {coef_subs}, qfj, qfj -> fj', ws, coef, phir, phir, optimize=True)
        else:
            raise ValueError(f'coef type {type(coef)} is not supported.')

        cell2dof = space.cell_to_dof()

        if out is None:
            F = np.zeros(gdof, dtype=mesh.ftype)
            np.add.at(F, cell2dof[face2cell[:, 0]], Al)
            np.add.at(F, cell2dof[face2cell[in_face_flag, 1]], Ar)
            return F
        else:
            np.add.at(out, cell2dof[face2cell[:, 0]], Al)
            np.add.at(out, cell2dof[face2cell[in_face_flag, 1]], Ar)
            return out
