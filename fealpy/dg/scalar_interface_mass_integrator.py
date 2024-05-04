
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray

from .utils import to_global
from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalerInterfaceMassIntegrator():
    r"""Scalar interface mass integrator."""
    def __init__(self, c: Union[NDArray, float, None]=None, q: Optional[int]=None) -> None:
        self.q = q
        self.coef = c

    def assembly_face_matrix(self, space: ScaledMonomialSpace, out=None):
        q = self.q or space.p + 1
        coef = self.coef
        mesh: PolygonMesh = space.mesh
        gdof = space.number_of_global_dofs()

        face2cell = mesh.ds.face_to_cell()
        in_face_flag = face2cell[:, 0] != face2cell[:, 1]
        fm = mesh.entity_measure('face')
        fm_in = fm[in_face_flag]
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.face_bc_to_point(bcs) #(NQ, bd_NF, GD)
        NQ, NF, GD = ps.shape

        phil = space.basis(ps, index=face2cell[:, 0]) # (NQ, NF, ldof)
        phir = space.basis(ps[:, in_face_flag, :], index=face2cell[in_face_flag, 1]) # (NQ, in_NF, ldof)

        if coef is None:
            All = np.einsum('q, qfi, qfj, f -> fij', ws, phil, phil, fm, optimize=True) # (NQ, ldof, ldof)
            Arr = np.einsum('q, qfi, qfj, f -> fij', ws, phir, phir, fm_in, optimize=True)
            Alr = -np.einsum('q, qfi, qfj, f -> fij', ws, phil[:, in_face_flag, :], phir, fm_in, optimize=True)
            Arl = -np.einsum('q, qfi, qfj, f -> fij', ws, phir, phil[:, in_face_flag, :], fm_in, optimize=True)
        elif np.isscalar(coef):
            All = np.einsum('q, qfi, qfj, f -> fij', ws, phil, phil, fm, optimize=True) * coef # (NQ, ldof, ldof)
            Arr = np.einsum('q, qfi, qfj, f -> fij', ws, phir, phir, fm_in, optimize=True) * coef
            Alr = -np.einsum('q, qfi, qfj, f -> fij', ws, phil[:, in_face_flag, :], phir, fm_in, optimize=True) * coef
            Arl = -np.einsum('q, qfi, qfj, f -> fij', ws, phir, phil[:, in_face_flag, :], fm_in, optimize=True) * coef
        elif isinstance(coef, np.ndarray):
            if coef.shape == (NF, ):
                coef_subs = 'f'
                coef_in = coef[in_face_flag]
            elif coef.shape == (NQ, NF):
                coef_subs = 'qf'
                coef_in = coef[:, in_face_flag]
            else:
                raise ValueError(f'coef.shape = {coef.shape} is not supported.')
            All = np.einsum(f'q, {coef_subs}, qfi, qfj, f -> fij', ws, coef, phil, phil, fm, optimize=True) # (NQ, ldof, ldof)
            Arr = np.einsum(f'q, {coef_subs}, qfi, qfj, f -> fij', ws, coef_in, phir, phir, fm_in, optimize=True)
            Alr = -np.einsum(f'q, {coef_subs}, qfi, qfj, f -> fij', ws, coef_in, phil[:, in_face_flag, :], phir, fm_in, optimize=True)
            Arl = -np.einsum(f'q, {coef_subs}, qfi, qfj, f -> fij', ws, coef_in, phir, phil[:, in_face_flag, :], fm_in, optimize=True)
        else:
            raise ValueError(f'coef type {type(coef)} is not supported.')

        face2celldof_left = space.cell_to_dof()[face2cell[:, 0]]
        face2celldof_right = space.cell_to_dof()[face2cell[:, 1]]

        R = to_global(All, face2celldof_left, face2celldof_left, gdof)
        R += to_global(Arl, face2celldof_right[in_face_flag], face2celldof_left[in_face_flag], gdof)
        R += to_global(Alr, face2celldof_left[in_face_flag], face2celldof_right[in_face_flag], gdof)
        R += to_global(Arr, face2celldof_right[in_face_flag], face2celldof_right[in_face_flag], gdof)

        if out is None:
            return R.tocsr()
        else:
            out += R.tocsr()
