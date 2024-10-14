import numpy as np

import jax
import jax.numpy as jnp
from scipy.sparse import csr_matrix

class ScalarInteriorPenaltyIntegrator:

    def __init__(self, q=None, gamma=1):
        self.q = q
        self.gamma = gamma

    def assembly_face_matrix(self, space, index=jnp.s_[:]):
        """
        @brief 计算三角形网格上的边 内罚 矩阵
        """

        mesh = space.mesh
        assert type(mesh).__name__ == "TriangleMesh"
        em   = mesh.entity_measure('edge')
        
        isBdEdge    = mesh.ds.boundary_edge_flag() 
        isInnerEdge = ~isBdEdge 

        p = space.p
        q = self.q if self.q is not None else p+3 

        ie2cd = space.dof.iedge2celldof
        be2cd = space.dof.bedge2celldof

        qf = mesh.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()
        
        gnjphi  = -space.grad_normal_jump_basis(bcs)
        gn2jphi = space.grad_normal_2_jump_basis(bcs)
        
        # 一阶法向导数矩阵
        P1 = jnp.einsum('q, qfi, qfj->fij', ws, gnjphi, gnjphi)
        P1 = P1*self.gamma

        P2 = jnp.einsum('q, qfi, qfj, f->fij', ws, gnjphi, gn2jphi, em[isInnerEdge])
        P2T = jnp.transpose(P2, axes=(0, 2, 1))

        P = (P2 + P2T) + P1
         
        I = jnp.broadcast_to(ie2cd[:, :, None], shape=P.shape)
        J = jnp.broadcast_to(ie2cd[:, None, :], shape=P.shape)

        gdof = space.dof.number_of_global_dofs()
        P = csr_matrix((P.flatten(), (I.flatten(), J.flatten())), shape=(gdof, gdof))

        # 边界的积分
        gnjphi  = -space.boundary_edge_grad_normal_jump_basis(bcs)
        gn2jphi = space.boundary_edge_grad_normal_2_jump_basis(bcs)

        P1 = jnp.einsum('q, qfi, qfj->fij', ws, gnjphi, gnjphi)
        P1 = P1*self.gamma

        P2 = jnp.einsum('q, qfi, qfj, f->fij', ws, gnjphi, gn2jphi, em[isBdEdge])
        P2T = jnp.transpose(P2, axes=(0, 2, 1))

        PP = (P2 + P2T) + P1
        
        I = jnp.broadcast_to(be2cd[:, :, None], shape=PP.shape)
        J = jnp.broadcast_to(be2cd[:, None, :], shape=PP.shape)

        gdof = space.dof.number_of_global_dofs()
        P = P+csr_matrix((PP.flatten(), (I.flatten(), J.flatten())), shape=(gdof, gdof))

        return P



