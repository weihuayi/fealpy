import pytest

from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.decorator import cartesian
from fealpy.levelset.ls_fem_solver import LSFEMSolver

import numpy as np


# A fixture for setting up the LSSolver environment with a predefined mesh and space.
@pytest.fixture
def ls_fem_solver_setup():
    # Create a mesh in a unit square domain with specified number of divisions along x and y.
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=100, ny=100)
    # Define a Lagrange finite element space of first order on the mesh.
    space = LagrangeFESpace(mesh, p=1)

    # Instantiate the LSFEMSolver with the space.
    solver = LSFEMSolver(space)

    return solver, space


# Test the initialization of the LSFEMSolver class.
def test_LSFEMSolver_init(ls_fem_solver_setup):
    solver, space = ls_fem_solver_setup

    # Check that the M matrix and space have been correctly initialized.
    assert solver.M is not None, "Matrix M should be initialized."
    assert solver.space is space, "The space attribute should be equal to the space provided."
