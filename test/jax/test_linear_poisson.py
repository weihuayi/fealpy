import pytest

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import TriangleMesh
from fealpy.jax import logger


def test_linear_poisson():
    logger.debug("Building the pde and the mesh!")
    pde = CosCosData()
    domain = pde.domain()
    mesh = TriangleMesh.from_box(box = domain, nx = 10, ny = 10)
    logger.debug(f"mesh with {mesh.number_of_nodes()} nodes and {mesh.number_of_cells()} cells.")
    logger.debug("Finish!")



if __name__ == "__main__":
    test_linear_poisson()



