
import jax
import jax.numpy as jnp


class ScalarLapalceIntegrator:

    def __init__(self, q=None):
        self.q = q

    def assembly_cell_matrix(self, space, index=jnp.s_[:]):
        """
        """
        p = space.p
        q = self.q if self.q is not None else p+1 

        qf = mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        NQ = len(ws)
