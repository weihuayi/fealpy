import numpy as np

import jax
import jax.numpy as jnp
from functools import partial
from scipy.sparse import csr_matrix

class ScalarBiharmonicIntegrator:

    def __init__(self, q=None):
        self.q = q

    def assembly_cell_matrix(self, space, index=jnp.s_[:]):
        """
        @brief 计算三角形网格上的单元 hessian 矩阵
        """

        mesh = space.mesh
        assert type(mesh).__name__ == "TriangleMesh"

        p = space.p
        q = self.q if self.q is not None else p+3 

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        R = space.hess_basis(bcs)
        M = jnp.einsum('q, qikl, qjrs->ijklrs', ws, R, R) 

        # 计算与单元相关的部分
        cm = mesh.entity_measure(index=index)
        glambda = mesh.grad_lambda(index=index)

        A = jnp.einsum('c, ckm, cln, crm, csn->cklrs', cm, glambda, glambda, glambda, glambda)

        # 计算 hessian 部分的刚度矩阵
        A = jnp.einsum('ijklrs, cklrs->cij', M, A)

        
        hphi = space.hess_basis(bcs, variable='x')
        A = jnp.einsum('c, q, cqlij, cqmij->clm', cm, ws, hphi, hphi)
        return A

