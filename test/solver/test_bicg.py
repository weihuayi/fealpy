from fealpy.backend import backend_manager as bm
from fealpy.solver import bicg as fealpy_bicg
from fealpy.sparse.csr_tensor import CSRTensor

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


class TestMINRESSolver:
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
    

    @pytest.mark.parametrize("backend", ['numpy', 'torch'])
    def test_bicg_cpu(self, backend):
        bm.set_backend(backend)
        
        # A, f：Matrix construction
        nx_1 = ny_1 = 200
        p = 1
        mesh_1 = TriangleMesh.from_box(domain, nx_1, ny_1)
        space = LagrangeFESpace(mesh_1, p=p)
        uh = space.function()
        uh_1 = space.function()

        A, f = self.get_Af(mesh_1, p=p)
        print(A.shape)

        if tmr is not None:
            tmr.send("Matrix construction completed.")
        
        # Solve using bicg in fealpy
        uh[:], info = fealpy_bicg(A, f, rtol=1e-8, atol=1e-8)

        if tmr is not None:
            tmr.send("fealpy_minres solver completed.")

        # Compute relative error
        err = mesh_1.error(pde.solution, uh)

        # Output results
        print('Iterations in fealpy:', info['niter'])
        print('Stop residual in fealpy:', info['residual'])
        print('Error in fealpy:', err)

        # Solve using bicg in scipy
        from scipy.sparse.linalg import bicg
        A = A.to_scipy()
        # M = M.to_scipy()
        if tmr is not None:
            tmr.send("A 转换为scipy.")

        uh_1[:], iteration, test1= bicg(A, f, rtol=1e-8,atol=1e-8)
        if tmr is not None:
            tmr.send("minres solver completed.")
        
        print('Iterations in scipy:', iteration)
        print('Stop residual in scipy:', test1)
        print('Error in scipy:', mesh_1.error(pde.solution, uh_1))
        
        tmr.send(None)  # End timer
        
        # Check convergence
        rtol = 1e-4
        if info['residual'] <= rtol:
            print("Converged: True")
            converged = True
        else:
            print("Converged: False")
            converged = False

        # Assert convergence
        assert converged, f"bicg solver did not converge: stop_res = {info['residual']} > rtol = {rtol}"


    def test_bicg_gpu(self):
        bm.set_backend('pytorch')
        bm.set_default_device("cuda")

        # A, f
        nx_1 = ny_1 = 200
        p = 1
        mesh_1 = TriangleMesh.from_box(domain, nx_1, ny_1)
        space = LagrangeFESpace(mesh_1, p=p)
        uh = space.function()
    
        A, f = self.get_Af(mesh_1, p=p)
        A, f = self._get_gpu_data(A, f)
        print(A.shape)
        
        # M
        if tmr is not None:
            tmr.send("Matrix construction completed.")
        
        # Solve using bicg in fealpy
        uh[:], info = fealpy_bicg(A, f, rtol=1e-8, atol=1e-8)

        if tmr is not None:
            tmr.send("fealpy_minres solver completed.")

        # Compute relative error
        err = mesh_1.error(pde.solution, uh)

        # Output results
        print('Iterations in fealpy:', info['niter'])
        print('Stop residual:', info['residual'])
        print('Error in fealpy:', err)

        tmr.send(None)  # End timer
        
        # Check convergence
        rtol = 1e-4
        if info['residual'] <= rtol:
            print("Converged: True")
            converged = True
        else:
            print("Converged: False")
            converged = False

        # Assert convergence
        assert converged, f"MINRES solver did not converge: stop_res = {info['residual']} > rtol = {rtol}"


if __name__ == '__main__':
    test = TestMINRESSolver()
    test.test_bicg_cpu('numpy')
    test.test_bicg_gpu()