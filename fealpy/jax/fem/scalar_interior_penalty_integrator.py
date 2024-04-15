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

        p = space.p
        q = self.q if self.q is not None else p+3 

        # 组装罚项矩阵
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()
        TD = mesh.top_dimension()

        isEdgeDof = (mesh.multi_index_matrix(p, 2) == 0) 
        cell2edge = mesh.ds.cell2edge #TODO: 增加接口
        NEC = mesh.ds.localEdge.shape[0]

        edge2cell = mesh.ds.edge2cell

        cell2edgesign = jnp.zeros((NC, NEC), dtype=np.bool_)
        # 第 i 个单元的第 j 条边的全局边方向与在本单元中的局部方向不同
        cell2edgesign = cell2edgesign.at[(edge2cell[:, 0], edge2cell[:, 2])].set(True)

        ldof = space.number_of_local_dofs() # 单元上所有自由度的个数
        edof = space.number_of_local_dofs(doftype='edge') # 单元边上的自由度
        ndof = ldof - edof
        edge2dof = jnp.zeros((NE, edof + 2*ndof), dtype=int)
        
        qf = mesh.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()
        
        NQ = len(ws)

        n = mesh.edge_unit_normal()
        cell2dof = space.cell_to_dof()

        # 每个积分点、每个边、每个基函数法向导数
        val1 = jnp.zeros((NQ, NE, edof + 2*ndof), dtype=jnp.float_)
        # 每个积分点、每个边、每个基函数二阶法向导数
        val2 = jnp.zeros((NQ, NE, edof + 2*ndof), dtype=jnp.float_)
        
        edge2cell = mesh.ds.edge2cell
        isBdEdge = edge2cell[:, 0] == edge2cell[:, 1] # 边界边

        #  循环每个边
        for i in range(TD+1):
            lidx, = jnp.nonzero( cell2edgesign[:, i]) # 单元是全局边的左边单元
            ridx, = jnp.nonzero(~cell2edgesign[:, i]) # 单元是全局边的右边单元
            idx0, = jnp.nonzero( isEdgeDof[:, i]) # 在边上的自由度
            idx1, = jnp.nonzero(~isEdgeDof[:, i]) # 不在边上的自由度

            eidx = cell2edge[:, i] # 第 i 个边的全局编号
            edge2dof = edge2dof.at[(eidx[lidx, None], jnp.arange(edof, edof+ndof))].set(cell2dof[lidx[:, None], idx1]) 
            edge2dof = edge2dof.at[(eidx[ridx, None], jnp.arange(edof+ndof, edof+2*ndof))].set(cell2dof[ridx[:, None], idx1])

            # 边上的自由度按编号大小进行排序
            idx = jnp.argsort(cell2dof[:, isEdgeDof[:, i]], axis=1)
            edge2dof = edge2dof.at[(eidx, slice(0, edof))].set(cell2dof[:, isEdgeDof[:, i]][jnp.arange(NC)[:, None], idx])

            
            # 边上的积分点转化为体上的积分点
            b = jnp.insert(bcs, i, 0, axis=1)
            # 计算一阶法向导数 (NQ, NC, cdof)
            cval = jnp.einsum('iqjm, im->qij', space.grad_basis(b, variable='x'), n[cell2edge[:, i]])
            # 不在边上的自由度取值
            val1 = val1.at[(slice(None), eidx[ridx], slice(edof + ndof, edof + 2 * ndof))].set(+cval[:, ridx[:, None], idx1])
            val1 = val1.at[(slice(None), eidx[lidx], slice(edof, edof + ndof))].set(-cval[:, lidx[:, None], idx1])
            # 边上的自由度取值
            val1 = val1.at[(slice(None), eidx[ridx], slice(0, edof))].set(+cval[:, ridx[:, None], idx0[idx[ridx, :]]])
            val1 = val1.at[(slice(None), eidx[lidx], slice(0, edof))].set(-cval[:, lidx[:, None], idx0[idx[lidx, :]]])
            
            # 计算二阶法向导数
            R = space.hess_basis(b)

            glambda = mesh.grad_lambda()
            A = jnp.einsum('ckm, cln, qikl->qcimn', glambda, glambda, R)

            cval = jnp.einsum('qcimn, cm, cn-> qci', A, n[cell2edge[:, i]],
                n[cell2edge[:, i]])
            cval = cval/2.0
            
            val2 = val2.at[(slice(None), eidx[ridx], slice(edof + ndof, edof + 2
                * ndof))].set(+cval[:, ridx[:, None], idx1]) 
            val2 = val2.at[(slice(None), eidx[lidx], slice(edof, edof +
                ndof))].set(+cval[:, lidx[:, None], idx1]) 
            val2 = val2.at[(slice(None), eidx[ridx], slice(0, edof))].set(+cval[:,
                ridx[:, None], idx0[idx[ridx, :]]]) 
            val2 = val2.at[(slice(None), eidx[lidx], slice(0, edof))].set(
                    +cval[:, lidx[:, None], idx0[idx[lidx, :]]])
            
        val2 = val2.at[..., isBdEdge, :].mul(2.0)
        h = mesh.entity_measure('edge')
        
        # 一阶法向导数矩阵
        P1 = jnp.einsum('q, qfi, qfj->fij', ws, val1, val1)
        P1 = P1*self.gamma

        P2 = jnp.einsum('q, qfi, qfj, f->fij', ws, val1, val2, h)
        P2T = jnp.transpose(P2, axes=(0, 2, 1))
        P = (P2+P2T)/2.0 + P1
        
        I = jnp.broadcast_to(edge2dof[:, :, None], shape=P.shape)
        J = jnp.broadcast_to(edge2dof[:, None, :], shape=P.shape)

        gdof = space.dof.number_of_global_dofs()
        P = csr_matrix((P.flatten(), (I.flatten(), J.flatten())), shape=(gdof, gdof))
        return P



