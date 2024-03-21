
import numpy as np

import jax
import jax.numpy as jnp


class ScalarSourceIntegrator():

    def __init__(self, f, q=None):
        """
        @brief

        @param[in] f 
        """
        self.f = f
        self.q = q
        self.vector = None

    def assembly_cell_vector(self, space):
        """
        @brief 组装单元向量

        @param[in] space 一个标量的函数空间

        """
        f = self.f
        p = space.p
        q = self.q

        q = p+3 if q is None else q

        mesh = space.mesh
        assert type(mesh).__name__ == "TriangleMesh"
        cellmeasure = mesh.entity_measure('cell')

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        phi = space.basis(bcs) #TODO: 考虑非重心坐标的情形

        ps = mesh.bc_to_point(bcs, index=index)
        val = f(ps)

        bb = jnp.einsum('q, qc, qci, c->ci', ws, val, phi, cellmeasure, optimize=True)
        return bb 
        
