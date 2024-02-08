import pytest

import numpy as np

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
    tmesh = Mesh.from_box(box=domain, nx=10, ny=10)
    node = tmesh.entity('node')
    cell = tmesh.entity('cell')
    logger.debug(f"mesh with {mesh.number_of_nodes()} nodes and {mesh.number_of_cells()} cells.")
    logger.debug("Finish!")

    mesh = TriangleMesh(node, cell)
    space = LagrangeFESpace(mesh, p=2)


    qf = mesh.integrator(3)
    bcs, ws = qf.get_quadrature_points_and_weights()


    R = space.grad_basis(bcs, varialbes='u')

    glambda = mesh.grad_lambda()


    

    


if __name__ == "__main__":
    test_linear_poisson()



