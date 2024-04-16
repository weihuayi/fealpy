
from typing import Union, Optional

import numpy as np
from numpy.typing import NDArray

from .utils import to_global
from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalerInterfaceIntegrator():
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
        fn = mesh.face_unit_normal()
        fm = mesh.entity_measure('face')
        fm_in = fm[in_face_flag]
        qf = mesh.integrator(q, 'face')
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.face_bc_to_point(bcs) #(NQ, NF, GD)
        NQ, NF, GD = ps.shape
        phil = space.basis(ps, index=face2cell[:, 0]) # (NQ, NF, ldof)
        phir = space.basis(ps[:, in_face_flag, :], index=face2cell[in_face_flag, 1]) # (NQ, in_NF, ldof)
        gphil = np.sum(
            space.grad_basis(ps, index=face2cell[:, 0]) * fn[None, :, None, :],
            axis=-1
        ) # (NQ, NF, ldof)
        gphir = np.sum(
            space.grad_basis(ps[:, in_face_flag, :], index=face2cell[in_face_flag, 1])\
                * fn[None, in_face_flag, None, :],
            axis=-1
        ) # (NQ, in_NF, ldof)

        if coef is None:
            All = -np.einsum('q, qfi, qfj, f -> fij', ws, phil, gphil, fm, optimize=True) # (NF, ldof)
            Arr = np.einsum('q, qfi, qfj, f -> fij', ws, phir, gphir, fm_in, optimize=True) # (in_NF, ldof)
            Alr = -np.einsum('q, qfi, qfj, f -> fij', ws, phil[:, in_face_flag, :], gphir, fm_in, optimize=True)
            Arl = np.einsum('q, qfi, qfj, f -> fij', ws, phir, gphil[:, in_face_flag, :], fm_in, optimize=True)
        elif np.isscalar(coef):
            All = np.einsum('q, qfi, qfj, f -> fij', ws, phil, gphil, fm, optimize=True) * coef # (NF, ldof)
            Arr = -np.einsum('q, qfi, qfj, f -> fij', ws, phir, gphir, fm_in, optimize=True) * coef # (in_NF, ldof)
            Alr = np.einsum('q, qfi, qfj, f -> fij', ws, phil[:, in_face_flag, :], gphir, fm_in, optimize=True) * coef
            Arl = -np.einsum('q, qfi, qfj, f -> fij', ws, phir, gphil[:, in_face_flag, :], fm_in, optimize=True) * coef
        elif isinstance(coef, np.ndarray):
            if coef.shape == (NF, ):
                coef_subs = 'f'
            elif coef.shape == (NQ, NF):
                coef_subs = 'qf'
            else:
                raise ValueError(f'coef.shape = {coef.shape} is not supported.')
            All = -np.einsum(f'q, {coef_subs}, qfi, qfj, f -> fij', ws, coef, phil, gphil, fm, optimize=True) # (NF, ldof)
            Arr = np.einsum(f'q, {coef_subs}, qfi, qfj, f -> fij', ws, coef, phir, gphir, fm_in, optimize=True) # (in_NF, ldof)
            Alr = -np.einsum(f'q, {coef_subs}, qfi, qfj, f -> fij', ws, coef, phil[:, in_face_flag, :], gphir, fm_in, optimize=True)
            Arl = np.einsum(f'q, {coef_subs}, qfi, qfj, f -> fij', ws, coef, phir, gphil[:, in_face_flag, :], fm_in, optimize=True)
        else:
            raise ValueError(f'coef type {type(coef)} is not supported.')

        All[in_face_flag, ...] /= 2
        Arr /= 2
        Alr /= 2
        Arl /= 2

        face2celldof_left = space.cell_to_dof()[face2cell[:, 0]]
        face2celldof_right = space.cell_to_dof()[face2cell[:, 1]]

        R = to_global(All, face2celldof_left, face2celldof_left, gdof)
        R += to_global(Arl, face2celldof_right[in_face_flag], face2celldof_left[in_face_flag], gdof)
        R += to_global(Alr, face2celldof_left[in_face_flag], face2celldof_right[in_face_flag], gdof)
        R += to_global(Arr, face2celldof_right[in_face_flag], face2celldof_right[in_face_flag], gdof)
        R += R.transpose()

        if out is None:
            return R.tocsr()
        else:
            out += R.tocsr()
