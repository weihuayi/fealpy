import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

class ScalarBiharmonicIntegrator:

    def __init__(self, q=None):
        self.q = q

    def assembly_cell_matrix(self, space, index=jnp.s_[:]):
        """
        @brief 计算三角形网格上的单元 Laplace 矩阵
        """

        mesh = space.mesh
        assert type(mesh).__name__ == "TriangleMesh"

        p = space.p
        q = self.q if self.q is not None else p+3 

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        # 计算与单元无关的部分
        phi = space.basis(bcs) # (NQ, ldof)
        R = self.hessian(phi, bcs) # 计算 hessian矩阵
        M = jnp.einsum('q, qikm, qjlm->ijkl', ws, R, R) # TODO
        
        # 计算与单元相关的部分
        cm = mesh.entity_measure()
        glambda = mesh.grad_lambda()


        # 计算 hessian 部分的刚度矩阵


        # 组装罚项矩阵
        NC = mesh.number_of_cells()
        NE = mesh.number_of_edges()

        isEdgeDof = (mesh.multi_index_matrix(p) == 0) # TODO 这里该如何拿到边的自由度
        cell2edge = mesh.ds.cell_to_edge()
        cell2edgesign = mesh.ds.cell_to_edge_sign()

        ldof = space.number_of_local_dofs() # 单元上所有自由度的个数
        edof = space.number_of_local_dofs('edge') # 单元边上的自由度
        ndof = ldof - edof
        edge2dof = jnp.zeros((NF, edof + 2*ndof), dtype=jnp.int)
        NQ = len(ws)

        n = mesh.edge_unit_normal()
        cell2dof = space.cell_to_dof()
        # 每个积分点、每个边、每个基函数法向导数
        val = jnp.zeros((NQ, NE, edof + 2*ndof), dtype=jnp.float)
        TD = mesh.top_dimension()
        #  循环每个边
        for i in range(TD+1):
            lidx, = jnp.nonzero( cell2edgesign[:, i]) # 单元是全局边的左边单元
            ridx, = jnp.nonzero(~cell2edgesign[:, i]) # 单元是全局边的右边单元
            idx0, = jnp.nonzero( isEdgeDof[:, i]) # 在边上的自由度
            idx1, = jnp.nonzero(~isEdgeDof[:, i]) # 不在边上的自由度

            eidx = cell2edge[:, i] # 第 i 个边的全局编号
            edge2dof[eidx[lidx, None], jnp.arange(edof,      edof+  ndof)] = cell2dof[lidx[:, None], idx1]
            edge2dof[eidx[ridx, None], jnp.arange(edof+ndof, edof+2*ndof)] = cell2dof[ridx[:, None], idx1]

            # 边上的自由度按编号大小进行排序
            idx = jnp.argsort(cell2dof[:, isEdgeDof[:, i]], axis=1)
            edge2dof[eidx, 0:edof] = cell2dof[:, isEdgeDof[:, i]][jnp.arange(NC)[:, None], idx]

            # 边上的积分点转化为体上的积分点
            b = jnp.insert(bcs, i, 0, axis=1)
            # (NQ, NC, cdof)
            cval = jnp.einsum('qijm, im->qij', self.grad_basis(b), n[cell2edge[:, i]])
            val[:, eidx[ridx, None], jnp.arange(edof+ndof, edof+2*ndof)] = +cval[:, ridx[:, None], idx1]
            val[:, eidx[lidx, None], jnp.arange(edof,      edof+  ndof)] = -cval[:, lidx[:, None], idx1]

            val[:, eidx[ridx, None], jnp.arange(0, edof)] += cval[:, ridx[:, None], idx0[idx[ridx, :]]]
            val[:, eidx[lidx, None], jnp.arange(0, edof)] -= cval[:, lidx[:, None], idx0[idx[lidx, :]]]


        return A

    def hessian(self, f, x):
        hess = jax.jacobian(lambda x: jax.jacobian(f, argnums=0)(x), argnums=0)(x)
#        y, jac = self.value_and_jacfwd(f, x)
#        y, hess = self.value_and_jacfwd(jac, x)
        return hess

    '''
    def value_and_jacfwd(self, f, x):
        pushfwd = functools.partial(jax.jvp, f, (x, ))
        basis = jnp.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
        y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis, ))
        return y, jac
    '''
