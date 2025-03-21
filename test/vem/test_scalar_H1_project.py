
import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.functionspace import ConformingScalarVESpace2d
from fealpy.vem.vem.h1 import ScalarDiffusionIntegrator
from fealpy.mesh import PolygonMesh, TriangleMesh

class TestScalarDiffusionIntegrator:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", 0)
    def test_scalar_diffusion_integrator(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],1,1)
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        mesh = PolygonMesh.from_mesh(mesh)
        space = ConformingScalarVESpace2d(mesh, p=3)
        integrator = ScalarDiffusionIntegrator(1, q=4)
        #a = integrator.h1_left(space)
        #print(a)
        b = integrator.h1_right(space)
        PI1 = integrator.H1_project_matrix(space)
        #print(PI1[0].shape)
        D = integrator.dof_matrix(space)
        #print(D[0].shape)
        M = integrator.L2_left(space)
        R = integrator.L2_project_matrix(space)
        stab = integrator.stabilization(space)
        K = integrator.assembly(space)
        def source(p):
            x = p[..., 0]
            y = p[..., 1]
            pi = bm.pi
            val = 2*pi*pi*np.sin(pi*x)*np.sin(pi*y)
            return val
        f = integrator.source(space, source)
        print(f)
        from fealpy.vem.vem.bilinear_form import BilinearForm
        bform = BilinearForm(space)
        bform.add_integrator(integrator)
        import ipdb
        ipdb.set_trace()
        KK = bform.assembly()
        print(KK.to_dense())
        from fealpy.vem.vem.linear_form import LinearForm
        lform = LinearForm(space)
        from fealpy.vem.vem.scalar_source_integrator import ScalarSourceIntegrator
        def ff(p):
            x = p[..., 0]
            y = p[..., 1]
            pi = bm.pi
            val = 2*pi*pi*np.sin(pi*x)*np.sin(pi*y)
            return val
        integrator = ScalarSourceIntegrator(ff, q=4)
        lform.add_integrator(integrator)
        F = lform.assembly()


if __name__ == "__main__":
    t = TestScalarDiffusionIntegrator()
    t.test_scalar_diffusion_integrator('numpy', 0)
