from fealpy.backend import backend_manager as bm
from fealpy.solver import lgmres as fealpy_lgmres
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


class TestGMRESSolver:
    def get_Af(self, mesh, p):
        """
        Assemble the stiffness matrix and load vector for the problem.
        """
        space = LagrangeFESpace(mesh, p=p)

        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator(method='fast'))

        lform = LinearForm(space)
        lform.add_integrator(ScalarSourceIntegrator(pde.source))

        A_without_apply = bform.assembly()
        F_without_apply = lform.assembly()

        A1, f = DirichletBC(space, gd=pde.solution).apply(A_without_apply, F_without_apply)
        A = A1.tocsr()  # Convert to compressed sparse row format
        return A, f


    def _get_gpu_data(self, A, f):
        A = A.device_put('cuda')
        return A, f
    
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    def test_lgmres_cpu(self, backend):
        bm.set_backend(backend)

        # A, f
        nx_1 = ny_1 = 300
        p = 1
        mesh_1 = TriangleMesh.from_box(domain, nx_1, ny_1)
        space = LagrangeFESpace(mesh_1, p=p)
        uh = space.function()
        uh_1 = space.function()
        
        A, f = self.get_Af(mesh_1, p=p)
        print(A.shape)
        if tmr is not None:
            tmr.send("Matrix assembly")

        # Solve using gmres in fealpy
        uh[:], info = fealpy_lgmres(A, f, atol=1e-12, rtol=1e-12, inner_m= 30, outer_k= 3, maxit=500)
        
        if tmr is not None:
            tmr.send("fealpy_gmres solving")
        
        err = mesh_1.error(pde.solution, uh)
        res_0 = bm.linalg.norm(f)
        stop_res = info['residual'] / res_0

        # Output iteration info
        print('Iterations (fealpy):', info['niter'])
        print('Error (fealpy):', err)
        print('Relative residual (fealpy):', stop_res)

        # Solve using gmres in scipy
        from scipy.sparse.linalg import lgmres
        A = A.to_scipy()
        
        if tmr is not None:
            tmr.send("Convert A to SciPy format")

        uh_1[:], info_1 = lgmres(A, f, atol=1e-12, rtol=1e-12, inner_m= 30, outer_k= 3, maxiter=500)
        if tmr is not None:
            tmr.send("SciPy gmres solving")
        
        print('converged (SciPy):',info_1)
        print('Error (SciPy):', mesh_1.error(pde.solution, uh_1))
        # print('Iterations (SciPy):', iteration)
        # print('Relative residual (SciPy):', r_norm / res_0)
        tmr.send(None)

        # Check convergence
        rtol = 1e-4
        converged = stop_res <= rtol
        print(f"Converged: {converged}")

        # Assert convergence
        assert converged, f"GMRES did not converge: residual = {stop_res:.2e} > rtol = {rtol}"
        
       
    def test_lgmres_gpu(self):
        bm.set_backend("pytorch")
        bm.set_default_device("cuda")
        
        # A, f
        nx_1 = ny_1 = 300
        p = 1
        mesh_1 = TriangleMesh.from_box(domain, nx_1, ny_1)
        space = LagrangeFESpace(mesh_1, p=p)
        uh = space.function()
        
        A, f = self.get_Af(mesh_1, p=p)
        A, f = self._get_gpu_data(A, f)
        print(A.shape)
        if tmr is not None:
            tmr.send("Matrix assembly")

        # Solve using gmres in fealpy
        uh[:], info = fealpy_lgmres(A, f, atol=1e-12, rtol=1e-12, inner_m= 30, outer_k= 3, maxit=500)
        if tmr is not None:
            tmr.send("fealpy_gmres solving")
        
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
    test.test_lgmres_cpu('numpy')
    # test.test_lgmres_gpu()