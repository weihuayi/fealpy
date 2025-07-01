
import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import ConformingScalarVESpace2d
from fealpy.vem import ScalarDiffusionIntegrator
from fealpy.mesh import PolygonMesh, TriangleMesh
from scalar_H1_project_data import *

class TestScalarDiffusionIntegrator:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", diff)
    def test_scalar_diffusion_integrator(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],1,1)
        NC = mesh.number_of_cells()
        #mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        #mesh = PolygonMesh.from_mesh(mesh)
        space = ConformingScalarVESpace2d(mesh, p=3)
        integrator = ScalarDiffusionIntegrator(1, q=4)
        #a = integrator.h1_left(space)
        #print(a)
        #b = integrator.h1_right(space)
        #PI1 = integrator.H1_project_matrix(space)
        #D = integrator.dof_matrix(space)
        #print('D',D)
        #M = integrator.L2_left(space)
        #R = integrator.L2_project_matrix(space)
        #stab = integrator.stabilization(space)
    #3print('R', R)
        #print('stab', stab)
        K = integrator.assembly(space)
        #for i in range(NC):
        #    np.testing.assert_allclose(K[i], data['diff'][i], atol=1e-10 )

        from fealpy.vem import BilinearForm
        bform = BilinearForm(space)
        bform.add_integrator(integrator)
        KK = bform.assembly()
        np.testing.assert_allclose(data['diff'], KK.to_dense(), atol=1e-10)
        

        from fealpy.vem.linear_form import LinearForm
        lform = LinearForm(space)
        from fealpy.vem.scalar_source_integrator import ScalarSourceIntegrator
        def ff(p):
            x = p[..., 0]
            y = p[..., 1]
            pi = bm.pi
            val = 2*pi*pi*np.sin(pi*x)*np.sin(pi*y)
            return val
        integrator = ScalarSourceIntegrator(ff, q=4)
        lform.add_integrator(integrator)
        F = lform.assembly()
        np.testing.assert_allclose(data['source'], F, atol=1e-10)


if __name__ == "__main__":
    t = TestScalarDiffusionIntegrator()
    t.test_scalar_diffusion_integrator('numpy', diff[0])
    t.test_scalar_diffusion_integrator('pytorch', diff[0])
