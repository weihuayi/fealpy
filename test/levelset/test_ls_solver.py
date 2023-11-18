import pytest
import numpy as np

from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.decorator import cartesian
from fealpy.levelset.ls_solver import LSSolver 


# A fixture for setting up the LSSolver environment and data
@pytest.fixture
def ls_solver_setup():
    # Create a mesh in a unit square domain with specified number of divisions along x and y.
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx = 128, ny = 128)
    # Define a Lagrange finite element space of first order on the mesh.
    space = LagrangeFESpace(mesh, p=1)

    @cartesian
    def circle_phi(p):
        x = p[...,0]
        y = p[...,1]
        val = np.sqrt((x - 0.5)**2 + (y - 0.75)**2) - 0.15
        return val

    phi = space.interpolate(circle_phi)

    return space, phi


def test_check_gradient_norm_at_interface(ls_solver_setup):
    space, phi = ls_solver_setup

    solver = LSSolver(space)

    diff_avg, diff_max = solver.check_gradient_norm_at_interface(phi)

    assert np.abs(diff_avg) < 0.01, "Average difference is too large"
    assert np.abs(diff_max) < 0.02, "Maximum difference is too large"


def test_solve_system(ls_solver_setup):
    # Unpack the solver from the setup fixture.
    space, _ = ls_solver_setup
    solver = LSSolver(space)

    from scipy.sparse import csr_matrix

    # Create a simple linear system with an identity matrix and a random vector b.
    size = 100
    A = csr_matrix(np.eye(size))
    b = np.random.rand(size)

    # Call the solve_system method which should solve the linear system A*x = b.
    result = solver.solve_system(A, b)
    
    # Assert that the result should be close to b since A is an identity matrix.
    assert np.allclose(result, b), "The solution should be close to b."


def test_compute_zero_level_set_area(ls_solver_setup):
    space, phi = ls_solver_setup
    solver = LSSolver(space)

    computed_area = solver.compute_zero_level_set_area(phi)

    radius = 0.15
    theoretical_area = np.pi * radius**2
    print(computed_area - theoretical_area)

    assert np.isclose(computed_area, theoretical_area, atol=0.001), f"Computed area ({computed_area}) does not match theoretical area ({theoretical_area})"
