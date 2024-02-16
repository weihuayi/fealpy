import pytest
import time

import numpy as np
from scipy.sparse import csr_array

import jax
import jax.numpy as jnp

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import TriangleMesh as Mesh

from fealpy.jax import logger
from fealpy.jax.functionspace import LagrangeFESpace


def kernel(R, ws):
    pass

def test_linear_poisson():
    logger.debug("Building the pde and the mesh!")
    pde = CosCosData()
    domain = pde.domain()
    tmesh = Mesh.from_box(box=domain, nx=1, ny=1)
    node = tmesh.entity('node')
    cell = tmesh.entity('cell')
    logger.debug(f"mesh with {mesh.number_of_nodes()} nodes and {mesh.number_of_cells()} cells.")
    logger.debug("Finish!")

    mesh = TriangleMesh(node, cell)
    space = LagrangeFESpace(mesh, p=2)

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


    

    


if __name__ == "__main__":
    test_linear_poisson()



