from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from ..quadrature import GaussLobattoQuadrature
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod,
    CoefLike
)


class DivDivIntegrator(LinearInt, OpInt, CellInt):
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        self.coef = coef
        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def assembly(self, space):
        scalar_space = space.scalar_space
        SM = scalar_space.SM
        PI1 = scalar_space.PI1
        NC = space.mesh.number_of_cells()

        coeff = self.coef
        Px, Py = scalar_space.smspace.partial_matrix()
        f = lambda x: x[0].T @ x[2].T @ x[1] @ x[2] @ x[0] 
        K00 = list(map(f, zip(PI1, SM, Px, Py)))
        f = lambda x: x[0].T @ x[2].T @ x[1] @ x[3] @ x[0] 
        K01 = list(map(f, zip(PI1, SM, Px, Py)))
        f = lambda x: x[0].T @ x[3].T @ x[1] @ x[3] @ x[0] 
        K11 = list(map(f, zip(PI1, SM, Px, Py)))
        f = lambda x: x[0].T @ x[3].T @ x[1] @ x[2] @ x[0] 
        K10 = list(map(f, zip(PI1, SM, Px, Py)))

        VK = []
        ldof = scalar_space.number_of_local_dofs() 
        for i in range(NC):
            K = bm.zeros((2*ldof[i], 2*ldof[i]), **space.mesh.fkwargs)
            K[:ldof[i], :ldof[i]] = coeff*K00[i] 
            K[:ldof[i], ldof[i]:] = coeff*K01[i]
            K[ldof[i]:, :ldof[i]] = coeff*K01[i].T
            K[ldof[i]:, ldof[i]:] = coeff*K11[i]
            VK.append(K)
        return VK
 
