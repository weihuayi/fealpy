
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray

from ..mesh import PolygonMesh
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d

ScaledMonomialSpace = ScaledMonomialSpace2d


class ScalarSourceIntegrator():
    r"""Scalar source integrator.

        $$(v, f)_\Omega$$
    """
    def __init__(self, source, c: Union[float, NDArray, None]=None, q: Optional[int]=None) -> None:
        self.source = source
        self.q = q
        self.coef = c

    def assembly_cell_vector(self, space: ScaledMonomialSpace, out: Optional[NDArray]=None, **kwargs):
        mesh: PolygonMesh = space.mesh
        coef = self.coef
        q = self.q or space.p + 1

        def func(x, index):
            gval = self.source(x) # TODO:
            phi = space.basis(x, index=index) # (NQ, NC, ldof)
            NQ, NC, _ = phi.shape

            if coef is None:
                return np.einsum('qc, qcj -> qcj', gval, phi)
            elif np.isscalar(coef):
                return np.einsum('qc, qcj -> qcj', gval, phi) * coef
            elif isinstance(coef, np.ndarray):
                if coef.shape == (NC, ):
                    coef_subs = 'c'
                elif coef.shape == (NQ, NC):
                    coef_subs = 'qc'
                else:
                    raise ValueError(f'coef.shape = {coef.shape} is not supported.')
                return np.einsum(f'{coef_subs}, qc, qcj -> qcj', coef, gval, phi)
            else:
                raise ValueError(f'coef type {type(coef)} is not supported.')

        result = mesh.integral(func, q, celltype=True) # (NC, ldof)
        if out is None:
            return result
        else:
            out[:] = result
            return out
