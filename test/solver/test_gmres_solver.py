from fealpy.backend import backend_manager as bm
from fealpy.solver import gmres

import pytest
from fealpy.utils import timer
from fealpy import logger

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
    BilinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    LinearForm,
    DirichletBC
)

logger.setLevel('INFO')

tmr = timer()
next(tmr)

# Define PDE model
pde = CosCosData()
domain = pde.domain()

def get_Af(mesh, p):
    """
    Assemble the stiffness matrix and load vector for the problem.
    """
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


class TestGMRESSolver:
    """
    Test case for GMRES solver.
    """
    @pytest.mark.parametrize("backend", ['numpy', 'torch'])
    def test_gmres(self, backend):
        bm.set_backend(backend)
        
        # Generate A, f, x
        nx_1 = ny_1 = 200
        p = 1
        mesh_1 = TriangleMesh.from_box(domain, nx_1, ny_1)
        space = LagrangeFESpace(mesh_1, p=p)
        uh = space.function()
        uh_1 = space.function()

        A, f = get_Af(mesh_1, p=p)
        print(A.shape)
        if tmr is not None:
            tmr.send("Matrix assembly")

        # Solve using gmres
        uh[:], info = gmres(A, f, atol=1e-8, rtol=1e-8, restart=20)
        if tmr is not None:
            tmr.send("gmres solving")

        err = mesh_1.error(pde.solution, uh)
        res_0 = bm.linalg.norm(f)
        stop_res = info['residual'] / res_0

        # Output iteration info
        print('Iterations (fealpy):', info['niter'])
        print('Error (fealpy):', err)
        print('Relative residual (fealpy):', stop_res)
        
        tmr.send(None)

        # Check convergence
        rtol = 1e-4
        converged = stop_res <= rtol
        print(f"Converged: {converged}")

        # Assert convergence
        assert converged, f"GMRES did not converge: residual = {stop_res:.2e} > rtol = {rtol}"

        
if __name__ == '__main__':
    test = TestGMRESSolver() 
    test.test_gmres('numpy')
    #test.test_gmres('pytorch')