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


def test_LSFEMSolver_reinit(ls_fem_solver_setup):
    space, u, phi0 = ls_fem_solver_setup

    # 提供一个非符号距离函数 phi0
    @cartesian
    def non_sdf(p):
        x = p[..., 0]
        y = p[..., 1]
        return (x - 0.5)**2 + (y - 0.5)**2  # 一个非符号距离函数的平方形式

    phi0 = space.interpolate(non_sdf)

    solver = LSSolver(space, phi0, u)

    diff_avg, diff_max = solver.check_gradient_norm(phi = phi0)
    print("diff_avg_0:", diff_avg)
    print("diff_max_0:", diff_max)

    # Instantiate the solver with the velocity field.
    solver_with_u = LSFEMSolver(space, u=u)

    # 执行重置化
    phi1 = solver_with_u.reinit(phi0)

    # Call the check_gradient_norm method which calculates the average and maximum difference from 1.
    diff_avg, diff_max = solver.check_gradient_norm(phi = phi1)
    print("diff_avg_1:", diff_avg)
    print("diff_max_1:", diff_max)

    # Assert that the average and maximum gradient norm differences should be close to 0.
    # This means that the gradient norm is close to 1 as expected for a signed distance function.
    assert np.isclose(diff_avg, 0, atol=1e-2), "The average difference should be close to 0."
    assert np.isclose(diff_max, 0, atol=1e-2), "The maximum difference should be close to 0."


