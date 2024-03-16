import numpy as np
import jax 
import jax.numpy as jnp

class ScalarDiffusionIntegrator:
    def assembly_cell_matrix(self, space, index=jnp.s_[:], cellmeasure):

        mesh = space.mesh
        cm = mesh.entity_measure()

        qf = mesh.integrator(3)
        bcs, ws = qf.get_quadrature_points_and_weights()


        R = space.grad_basis(bcs, varialbes='u') # (NQ, ldof, TD+1)

        M = jnp.enisum('q, qik, qjl->ijkl', ws, R, R)

        glambda = mesh.grad_lambda()

        A = jnp.enisum('ijkl, ckm, clm->cij', M, glambda, glambda, cm)

        cell2dof = space.cell_to_dof()
        I = jnp.broadcast_to(cell2dof[:, :, None], shape=A.shape)
        J = jnp.broadcast_to(cell2dof[:, None, :], shape=A.shape)
