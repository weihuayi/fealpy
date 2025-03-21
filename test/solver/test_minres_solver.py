from fealpy.backend import backend_manager as bm
from fealpy.solver import minres

import pytest
from fealpy.utils import timer
from fealpy import logger

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
    BilinearForm, ScalarDiffusionIntegrator, LinearForm, DirichletBC, ScalarSourceIntegrator
)

logger.setLevel('INFO')

# Initialize timer for performance tracking
tmr = timer()
next(tmr)

# Define PDE model
pde = CosCosData()
domain = pde.domain()


def get_Af(mesh, p):
    """Assemble the stiffness matrix and load vector for the problem."""
    space = LagrangeFESpace(mesh, p=p)

    uh = space.function()
    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator(method='fast'))

    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(pde.source))

    A_without_apply = bform.assembly()
    F_without_apply = lform.assembly()

    A1, f = DirichletBC(space, gd=pde.solution).apply(A_without_apply, F_without_apply)
    A = A1.tocsr()  # Convert to compressed sparse row format
    return A, f


class TestMINRESSolver:
    """Test case for MINRES solver."""

    @pytest.mark.parametrize("backend", ['numpy', 'torch'])
    def test_minres(self, backend: str):
        """Test the MINRES solver with different backends (numpy, torch)."""
        bm.set_backend(backend)

        # Define mesh and function space
        nx_1 = ny_1 = 200
        p = 1
        mesh_1 = TriangleMesh.from_box(domain, nx_1, ny_1)
        node = mesh_1.entity('node')
        space = LagrangeFESpace(mesh_1, p=p)
        uh = space.function()
        uh_1 = space.function()

        A, f = get_Af(mesh_1, p=p)
        print(A.shape)

        if tmr is not None:
            tmr.send("Matrix construction completed.")

        # Solve using fealpy_minres
        A = A.to_scipy()  # Convert to SciPy format
        uh[:], info = minres(A, f, rtol=1e-8)

        if tmr is not None:
            tmr.send("fealpy_minres solver completed.")

        tmr.send(None)  # End timer

        # Compute relative error
        err = mesh_1.error(pde.solution, uh)

        # Output results
        print('Iterations in fealpy:', info['niter'])
        print('Stop residual:', info['relative tolerance'])
        print('Error in fealpy:', err)

        # Check convergence
        rtol = 1e-4  # Convergence threshold
        if info['relative tolerance'] <= rtol:
            print("Converged: True")
            converged = True
        else:
            print("Converged: False")
            converged = False

        # Assert convergence
        assert converged, f"MINRES solver did not converge: stop_res = {info['relative tolerance']} > rtol = {rtol}"


if __name__ == '__main__':
    test = TestMINRESSolver()
    test.test_minres('pytorch')
    # test.test_minres('pytorch')