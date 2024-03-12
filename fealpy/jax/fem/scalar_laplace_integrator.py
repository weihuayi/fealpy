
import numpy as np
from fealpy import logger

import jax
import jax.numpy as jnp


class ScalarLapalceIntegrator:

    def __init__(self, q=None):
        self.q = q

    def assembly_cell_matrix(self, space, index=jnp.s_[:]):
        """
        """
        p = space.p
        q = self.q if self.q is not None else p+3 

        mesh = space.mesh
        qf = mesh.integrator(3, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        # 计算与单元无关的部分
        R = space.grad_basis(bcs, varialbes='u') # (NQ, ldof, TD+1)
        M = jnp.enisum('q, qik, qjl->ijkl', ws, R, R)

        # 计算与单元相关的部分
        cm = mesh.entity_measure()
        glambda = mesh.grad_lambda()

        # 计算最终的刚度矩阵
        A = jnp.enisum('c, ckm, clm, ijkl->cij', cm, glambda, glambda, M)

        return A

