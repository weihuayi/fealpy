
from typing import Union, Optional

import numpy as np
from numpy.typing import NDArray

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalerInterfaceIntegrator():
    def __init__(self, c: Union[NDArray, float, None]=None, q: Optional[int]=None) -> None:
        self.q = q
        self.coef = c

    def assembly_cell_matrix(self, space: ScaledMonomialSpace, out=None):
        q = self.q or space.p + 1
        coef = self.coef
        mesh: PolygonMesh = space.mesh
        NC = mesh.number_of_cells()
        ldof = space.number_of_local_dofs()

        face2cell = mesh.ds.face_to_cell()
        in_face_flag = face2cell[:, 0] != face2cell[:, 1]
        fn = mesh.face_unit_normal()
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
            Al = -np.einsum('q, qfi, qfj -> fij', ws, phil, gphil, optimize=True) # (NF, ldof)
            Ar = -np.einsum('q, qfi, qfj -> fij', ws, phir, gphir, optimize=True) # (in_NF, ldof)
        elif np.isscalar(coef):
            Al = -np.einsum('q, qfi, qfj -> fij', ws, phil, gphil, optimize=True) * coef # (NF, ldof)
            Ar = -np.einsum('q, qfi, qfj -> fij', ws, phir, gphir, optimize=True) * coef # (in_NF, ldof)
        elif isinstance(coef, np.ndarray):
            if coef.shape == (NF, ):
                coef_subs = 'f'
            elif coef.shape == (NQ, NF):
                coef_subs = 'qf'
            else:
                raise ValueError(f'coef.shape = {coef.shape} is not supported.')
            Al = -np.einsum(f'q, {coef_subs}, qfi, qfj -> fij', ws, coef, phil, gphil, optimize=True) # (NF, ldof)
            Ar = -np.einsum(f'q, {coef_subs}, qfi, qfj -> fij', ws, coef, phir, gphir, optimize=True) # (in_NF, ldof)
        else:
            raise ValueError(f'coef type {type(coef)} is not supported.')

        Al += Al.swapaxes(-1, -2)
        Al /= 2
        Ar += Ar.swapaxes(-1, -2)
        Ar /= 2

        if out is None:
            M = np.zeros((NC, ldof, ldof), dtype=mesh.ftype)
            np.add.at(M, face2cell[:, 0], -Al)
            np.add.at(M, face2cell[in_face_flag, 1], Ar)
            return M
        else:
            np.add.at(out, face2cell[:, 0], -Al)
            np.add.at(out, face2cell[in_face_flag, 1], Ar)
            return out
