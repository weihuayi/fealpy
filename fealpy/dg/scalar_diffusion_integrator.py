
from typing import Union, Optional
import numpy as np
from numpy.typing import NDArray

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalarDiffusionIntegrator():
    r"""Scalar diffusion integrator

        $$(\nabla v, \nabla u)_\Omega$$
    """
    def __init__(self, c: Union[float, NDArray, None]=None, q: Optional[int]=None) -> None:
        self.coef = c
        self.q = q

    def assembly_cell_matrix(self, space: ScaledMonomialSpace, out: Optional[NDArray]=None) -> NDArray:
        mesh: PolygonMesh = space.mesh
        coef = self.coef
        q = self.q or space.p + 1

        def func(x, index): # (NQ, SmallTri, GD)
            gphi = space.grad_basis(x, index=index, scaled=True) # (NQ, SmallTri, ldof, GD)
            NQ, NC, _, GD = gphi.shape

            if coef is None:
                return np.einsum('qcid, qcjd -> qcij', gphi, gphi)
            elif np.isscalar(coef):
                return np.einsum('qcid, qcjd -> qcij', gphi, gphi) * coef
            elif isinstance(coef, np.ndarray):
                if coef.shape == (NC, ):
                    coef_subs = 'c'
                elif coef.shape == (NQ, NC):
                    coef_subs = 'qc'
                else:
                    raise ValueError(f'coef.shape = {coef.shape} is not supported.')
                return np.einsum(f'{coef_subs}, qcid, qcjd -> qcij', coef, gphi, gphi)
            else:
                raise ValueError(f'coef type {type(coef)} is not supported.')

        result = mesh.integral(func, q, celltype=True) # (NC, ldof, ldof)
        if out is None:
            return result
        else:
            out[:] = result
            return out
