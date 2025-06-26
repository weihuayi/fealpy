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

    @assemblymethod('vector')
    def vector_assembly(self, space):

        from fealpy.vem.div_div_integrator import DivDivIntegrator
        DDv = DivDivIntegrator(coef=1,q=space.p+3)
        DDv = DDv.assembly(space)
        scalar_space = space.scalar_space
        SM = scalar_space.SM
        #KK = self.assembly(scalar_space)
        PI1 = scalar_space.PI1
        NC = space.mesh.number_of_cells()

        S = scalar_space.SS
        import ipdb
        ipdb.set_trace()
        f = lambda x: x[0].T @ x[1] @ x[0]
        KKK = list(map(f, zip(PI1, S)))
        h = scalar_space.smspace.cellsize
        stab = scalar_space.stab
        D = scalar_space.dof_matrix


        ldof = space.number_of_local_dofs()
        coeff = self.coef
        Px, Py = scalar_space.smspace.partial_matrix()
        f = lambda x: x[0].T @ x[2].T @ x[1] @ x[2] @ x[0] 
        K00 = list(map(f, zip(PI1, SM, Px, Py)))
        f = lambda x: x[0].T @ x[3].T @ x[1] @ x[2] @ x[0] 
        K01 = list(map(f, zip(PI1, SM, Px, Py)))
        f = lambda x: x[0].T @ x[3].T @ x[1] @ x[3] @ x[0] 
        K11 = list(map(f, zip(PI1, SM, Px, Py)))
        f = lambda x: x[0].T @ x[2].T @ x[1] @ x[3] @ x[0] 
        K10 = list(map(f, zip(PI1, SM, Px, Py)))
        NV = space.mesh.number_of_vertices_of_cells()

        ldof = scalar_space.number_of_local_dofs() 
        VK = []
        for i in range(NC):
            K = bm.zeros((2*ldof[i], 2*ldof[i]), **space.mesh.fkwargs)
            #K[:ldof[i], :ldof[i]] = coeff*K00[i] + coeff*KKK[i] + stab[i]
            #K[:ldof[i], :ldof[i]] = coeff*K00[i] + coeff*KKK[i] 
            K[:ldof[i], :ldof[i]] = 2*coeff*K00[i] + coeff*K11[i] 
            #+ 0.5*bm.trace(coeff*K00[i]+coeff*KKK[i]+DDv[i][:ldof[i],:ldof[i]])*(bm.eye(PI1[i].shape[1])-D[i]@PI1[i])
            K[:ldof[i], ldof[i]:] = coeff*K01[i]
            K[ldof[i]:, :ldof[i]] = coeff*K01[i].T
            #K[ldof[i]:, ldof[i]:] = coeff*K11[i] + coeff*KKK[i] + stab[i]
            K[ldof[i]:, ldof[i]:] = 2*coeff*K11[i] + coeff*K00[i] 
            #+ 0.5*bm.trace(coeff*K00[i]+coeff*KKK[i]+DDv[i][ldof[i]:,ldof[i]:])*(bm.eye(PI1[i].shape[1])-D[i]@PI1[i])
            #VVK = bm.zeros_like(K)
            #VVK[:ldof[i], :ldof[i]] = bm.eye(PI1[i].shape[1])-D[i]@PI1[i] 
            #VVK[ldof[i]:, ldof[i]:] = bm.eye(PI1[i].shape[1])-D[i]@PI1[i] 
            
            #K = K + 0.5*bm.trace(K+DDv[i])*VVK
            VK.append(K)

        return VK

   
        







        



