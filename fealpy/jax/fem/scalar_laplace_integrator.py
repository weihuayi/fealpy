
import numpy as np

import jax
import jax.numpy as jnp


class ScalarLaplaceIntegrator:

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
        R = space.grad_basis(bcs, variable='u') # (NQ, ldof, TD+1)
        M = jnp.einsum('q, qik, qjl->ijkl', ws, R, R)

        # 计算与单元相关的部分
        cm = mesh.entity_measure()
        glambda = mesh.grad_lambda()

        # 计算最终的刚度矩阵
        A = jnp.einsum('c, ckm, clm, ijkl->cij', cm, glambda, glambda, M)

        return A

