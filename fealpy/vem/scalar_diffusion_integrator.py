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


class ScalarDiffusionIntegrator(LinearInt, OpInt, CellInt):
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
#
#    @enable_cache
#    def fetch(self, space: _FS):
#        q = self.q
#        index = self.index
#        mesh = getattr(space, 'mesh', None)
#
#        if not isinstance(mesh, HomogeneousMesh):
#            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
#                               f"homogeneous meshes, but {type(mesh).__name__} is"
#                               "not a subclass of HomoMesh.")
#
#        cm = mesh.entity_measure('cell', index=index)
#        q = space.p+3 if self.q is None else self.q
#        qf = mesh.quadrature_formula(q, 'cell')
#        bcs, ws = qf.get_quadrature_points_and_weights()
#        phi = space.basis(bcs, index=index)
#        return bcs, ws, phi, cm, index

    @enable_cache
    def assembly(self, space):
        p = space.p
        S = space.SS
        PI1 = space.PI1
        f = lambda x: x[0].T @ x[1] @ x[0]
        K = list(map(f, zip(PI1, S)))
        stab = space.stab
        KK = list(map(lambda x: x[0] + x[1], zip(K, stab)))
        return KK

    
        







        



