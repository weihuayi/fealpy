import pytest
from fealpy.functionspace import LagrangeFESpace
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.decorator import cartesian
from fealpy.levelset.ls_solver import IterationCounter, LSSolver 
import numpy as np


# A fixture for setting up the LSSolver environment with a predefined mesh and space.
@pytest.fixture
def ls_solver_setup():
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

    # Instantiate the solver with the space and interpolated functions.
    solver = LSSolver(space, phi0, u)

    return solver, phi0, u


# Test the IterationCounter class.
def test_IterationCounter():
    # Initialize the IterationCounter with display set to False.
    counter = IterationCounter(disp = False)

    # Initially, the iteration count should be zero.
    assert counter.niter == 0

    # After calling the counter, the iteration count should increment.
    counter()
    assert counter.niter == 1


# Test the initialization of the LSSolver class.
def test_LSSolver_init(ls_solver_setup):
    # Unpack the solver, phi0 and u from the setup fixture.
    solver, phi0, u = ls_solver_setup

    # Check if the solver's phi0 and u attributes match the ones generated in the fixture.
    assert solver.phi0 is phi0
    assert solver.u is u


# Test the output method of the LSSolver class.
def test_LSSolver_output(ls_solver_setup, tmp_path):
    # Unpack the solver from the setup fixture.
    solver, _, _, = ls_solver_setup

    # Define a directory for output using pytest's temporary path fixture.
    output_dir = str(tmp_path)
    filename_prefix = "test_output"
    timestep = 0

    # Call the output method which should generate a file.
    solver.output(timestep, output_dir, filename_prefix)

    # Check if the file was created with the expected name.
    expected_file_path = tmp_path / f"{filename_prefix}_{timestep:010}.vtu"
    assert expected_file_path.is_file()


# Test the check_gradient_norm method of the LSSolver class.
def test_check_gradient_norm(ls_solver_setup):
    # Unpack the solver from the setup fixture.
    solver, _, _, = ls_solver_setup

    # Define a signed distance function for a circle for testing the gradient norm.
    @cartesian
    def signed_distance_circle(p):
        x = p[..., 0]
        y = p[..., 1]
        return np.sqrt(x**2 + y**2) - 1
    
    # Interpolate the function onto the finite element space.
    space = solver.space
    phi = space.interpolate(signed_distance_circle)

    # Call the check_gradient_norm method which calculates the average and maximum difference from 1.
    diff_avg, diff_max = solver.check_gradient_norm(phi)
    print("diff_avg:", diff_avg)
    print("diff_max:", diff_max)

    # Assert that the average and maximum gradient norm differences should be close to 0.
    # This means that the gradient norm is close to 1 as expected for a signed distance function.
    assert np.isclose(diff_avg, 0, atol=1e-6), "The average difference should be close to 0."
    assert np.isclose(diff_max, 0, atol=1e-6), "The maximum difference should be close to 0."


# Test the solve_system method of the LSSolver class.
def test_LSSolver_solve_system(ls_solver_setup):
    # Unpack the solver from the setup fixture.
    solver, _, _, = ls_solver_setup

    from scipy.sparse import csr_matrix

    # Create a simple linear system with an identity matrix and a random vector b.
    size = 10
    A = csr_matrix(np.eye(size))
    b = np.random.rand(size)

    # Call the solve_system method which should solve the linear system A*x = b.
    result = solver.solve_system(A, b)
    
    # Assert that the result should be close to b since A is an identity matrix.
    assert np.allclose(result, b), "The solution should be close to b."
