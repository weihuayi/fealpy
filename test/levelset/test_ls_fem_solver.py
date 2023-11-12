import pytest

from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.decorator import cartesian
from fealpy.levelset.ls_fem_solver import LSFEMSolver
from fealpy.levelset.ls_fem_solver import LSSolver

import numpy as np


# A fixture for setting up the LSSolver environment with a predefined mesh and space.
@pytest.fixture
def ls_fem_solver_setup():
    # Create a mesh in a unit square domain with specified number of divisions along x and y.
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=100, ny=100)
    # Define a Lagrange finite element space of first order on the mesh.
    space = LagrangeFESpace(mesh, p=1)

    # Define a velocity field function for testing.
    @cartesian
    def velocity_field(p):
        x = p[..., 0]
        y = p[..., 1]
        u = np.zeros(p.shape)
        u[..., 0] = np.sin(np.pi * x) ** 2 * np.sin(2 * np.pi * y)
        u[..., 1] = -np.sin(np.pi * y) ** 2 * np.sin(2 * np.pi * x)
        return u

    # Define a circle function representing the level set function for testing.
    @cartesian
    def circle(p):
        x = p[..., 0]
        y = p[..., 1]
        return np.sqrt((x - 0.5) ** 2 + (y - 0.75) ** 2) - 0.15

    # Interpolate the functions onto the finite element space.
    phi0 = space.interpolate(circle)
    u = space.interpolate(velocity_field, dim=2)

    return space, u, phi0


# Test the initialization of the LSFEMSolver class.
def test_LSFEMSolver_init_without_u(ls_fem_solver_setup):
    space, _, _ = ls_fem_solver_setup

    solver_without_u = LSFEMSolver(space)

    # Check that the mass matrix M is initialized.
    assert solver_without_u.M is not None, "Mass matrix M should be initialized."

    # Check that the space attribute matches the provided finite element space.
    assert solver_without_u.space is space, "The space attribute should match the provided finite element space."

    # Check that the velocity field u is None when not provided during initialization.
    assert solver_without_u.u is None, "Velocity field u should be None when not provided during initialization."

    # Check that the convection matrix C does not exist or is None when u is not provided.
    assert not hasattr(solver_without_u, 'C') or solver_without_u.C is None, "Convection matrix C should not exist or be None when velocity u is not provided."

def test_LSFEMSolver_init_with_u(ls_fem_solver_setup):
    space, u, _ = ls_fem_solver_setup

    # Instantiate the solver with the velocity field.
    solver_with_u = LSFEMSolver(space, u=u)

    # Check that the velocity field u is not None when provided during initialization.
    assert solver_with_u.u is not None, "Velocity field u should not be None when provided during initialization."
    # Check that the convection matrix C is initialized when u is provided.
    assert solver_with_u.C is not None, "Convection matrix C should be initialized when velocity u is provided."


def test_LSFEMSolver_solve(ls_fem_solver_setup):
    space, u, phi0 = ls_fem_solver_setup

    dt = 0.01  # A small time step

    # Instantiate the solver with the velocity field.
    solver_with_u = LSFEMSolver(space, u=u)

    # Perform one step of the level set evolution.
    phi1 = solver_with_u.solve(phi0, dt, u=u)

    # Check if the result is a numpy array, which it should be after solving.
    assert isinstance(phi1, np.ndarray), "The result of the solve method should be a numpy array."


def test_LSFEMSolver_reinit(ls_fem_solver_setup):
    space, u, _ = ls_fem_solver_setup

    # 提供一个非符号距离函数 phi0
    @cartesian
    def non_sdf(p):
        x = p[..., 0]
        y = p[..., 1]
        return (x - 0.5)**2 + (y - 0.5)**2  # 一个非符号距离函数的平方形式

    phi0 = space.interpolate(non_sdf)
    print("phi0:", phi0)

    # Instantiate the solver with the velocity field.
    solver_with_u = LSFEMSolver(space, u=u)

    solver = LSSolver(space, phi0, u)

    diff_avg, diff_max = solver.check_gradient_norm(phi = phi0)
    print("diff_avg_0:", diff_avg)
    print("diff_max_0:", diff_max)


    # 执行重置化
    phi1 = solver_with_u.reinit(phi0)
    print("phi1:", phi1)

    # Call the check_gradient_norm method which calculates the average and maximum difference from 1.
    diff_avg, diff_max = solver.check_gradient_norm(phi = phi1)
    print("diff_avg_1:", diff_avg)
    print("diff_max_1:", diff_max)

    # Assert that the average and maximum gradient norm differences should be close to 0.
    # This means that the gradient norm is close to 1 as expected for a signed distance function.
    assert np.isclose(diff_avg, 0, atol=1e-2), "The average difference should be close to 0."
    assert np.isclose(diff_max, 0, atol=1e-2), "The maximum difference should be close to 0."


