
from typing import Optional, Literal

from scipy.sparse import csr_matrix

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..decorator.variantmethod import variantmethod
from .integrator import LinearInt, OpInt, CellInt, enable_cache


class ScalarInteriorPenaltyIntegrator(LinearInt, OpInt, CellInt):
    r"""The interior penalty integrator for function spaces based on homogeneous meshes."""
    def __init__(self, coef: Optional[CoefLike] = None, q: Optional[int] = None, gamma: Optional[float] = 1, *,
                 index: Index = _S,
                 batched: bool = False,
                 method: Literal['fast', 'nonlinear', 'isopara', None] = None) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.gamma = gamma
        self.index = index
        self.batched = batched
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarDiffusionIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        em = mesh.entity_measure('edge', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()
        return bcs, ws, em

    @enable_cache
    def fetch_gnjphi(self, space: _FS):
        bcs = self.fetch(space)[0]
        return space.grad_normal_jump_basis(bcs)

    @enable_cache
    def fetch_ggnjphi(self, space: _FS):
        bcs = self.fetch(space)[0]
        return space.grad_grad_normal_jump_basis(bcs)
    
    @enable_cache
    def fetch_bgnjphi(self, space: _FS):
        bcs = self.fetch(space)[0]
        return space.boundary_edge_grad_normal_jump_basis(bcs)
    
    @enable_cache
    def fetch_bggnjphi(self, space: _FS):
        bcs = self.fetch(space)[0]
        return space.boundary_edge_grad_grad_normal_jump_basis(bcs)

    @variantmethod
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, em = self.fetch(space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='edge', index=self.index)
        gamma = self.gamma

        gnjphi = self.fetch_gnjphi(space)
        gn2jphi = self.fetch_ggnjphi(space)

        # 一阶法向导数矩阵
        P1 = bm.einsum('q, qfi, qfj->fij', ws, gnjphi, gnjphi)
        P1 = P1*self.gamma

        isBdEdge    = mesh.boundary_edge_flag() 
        isInnerEdge = ~isBdEdge 


        P2 = bm.einsum('q, qfi, qfj, f->fij', ws, gnjphi, gn2jphi, em[isInnerEdge])
        P2T = bm.permute_dims(P2, axes=(0, 2, 1))

        P = (P2 + P2T) + P1
        
        
        NC = mesh.number_of_cells()
        ie2cd = space.dof.iedge2celldof
        be2cd = space.dof.bedge2celldof
        
        I = bm.broadcast_to(ie2cd[:, :, None], P.shape)
        J = bm.broadcast_to(ie2cd[:, None, :], P.shape)

        gdof = space.dof.number_of_global_dofs()
        P = csr_matrix((P.flatten(), (I.flatten(), J.flatten())), shape=(gdof, gdof))

        # 边界的积分
        bgnjphi = -1 * self.fetch_bgnjphi(space)
        bggnjphi = self.fetch_bggnjphi(space)

        P1 = bm.einsum('q, qfi, qfj->fij', ws, bgnjphi, bgnjphi)
        P1 = P1*self.gamma

        P2  = bm.einsum('q, qfi, qfj, f->fij', ws, bgnjphi, bggnjphi, em[isBdEdge])
        P2T = bm.permute_dims(P2, axes=(0, 2, 1))

        PP = (P2+P2T) + P1
                
        I = bm.broadcast_to(be2cd[:, :, None], PP.shape)
        J = bm.broadcast_to(be2cd[:, None, :], PP.shape)

        gdof = space.dof.number_of_global_dofs()
        P = P+csr_matrix((PP.flatten(), (I.flatten(), J.flatten())), shape=(gdof, gdof))
        
        return P


    
    @assembly.register('fast')
    def fast_assembly(self, space: _FS) -> TensorLike:
        """
        限制：常系数、单纯形网格
        TODO: 加入 assert
        """
        pass
