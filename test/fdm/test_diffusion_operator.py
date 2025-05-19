from fealpy.fdm import DiffusionOperator
from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh
import numpy as np
import pytest
from diffusion_operator_data import A_1d, A_2d, A_3d

class TestDiffusionOperator:
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_1d_operator(self, backend):
        bm.set_backend(backend)
        diffusion_coef = bm.array([[1.0]], dtype=bm.float64)
        domain = [0, 1]
        nx = 4
        extent = [0, nx]
        mesh = UniformMesh(domain, extent)
        A = DiffusionOperator(mesh=mesh, diffusion_coef=diffusion_coef).assembly()
        A_test = bm.to_numpy(A.toarray())
        np.testing.assert_allclose(A_test, A_1d, atol=1e-10)

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_2d_operator(self, backend):
        bm.set_backend(backend)
        diffusion_coef = bm.array([[1.0, -1.0], [-2.0, 3.0]], dtype=bm.float64)
        domain = [0, 1, 0, 1]
        nx = 2
        ny = 2
        extent = [0, nx, 0, ny]
        mesh = UniformMesh(domain, extent)
        A = DiffusionOperator(mesh=mesh, diffusion_coef=diffusion_coef).assembly()
        A_test = bm.to_numpy(A.toarray())
        np.testing.assert_allclose(A_test, A_2d, atol=1e-10)

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_3d_operator(self, backend):
        bm.set_backend(backend)
        diffusion_coef = bm.array([[1.0, -1.0, 0.0], [-2.0, 3.0, 0.0], [0.0, 2.0, 4.0]], dtype=bm.float64)
        domain = [0, 1, 0, 1, 0, 1]
        nx = 1
        ny = 1
        nz = 1
        extent = [0, nx, 0, ny, 0, nz]
        mesh = UniformMesh(domain, extent)
        A = DiffusionOperator(mesh=mesh, diffusion_coef=diffusion_coef).assembly()
        A_test = bm.to_numpy(A.toarray())
        np.testing.assert_allclose(A_test, A_3d, atol=1e-10)

if __name__ == "__main__":
    pytest.main(["-s", "-v", "test_diffusion_operator.py"])
