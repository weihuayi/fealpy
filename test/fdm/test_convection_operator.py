from fealpy.fdm import ConvectionOperator
from fealpy.backend import backend_manager as bm
import numpy as np
from fealpy.mesh import UniformMesh
import pytest
from convection_operator_data import A0_1d, A1_1d, A0_2d, A1_2d, A0_3d, A1_3d

class TestConvectionOperator:
    @pytest.mark.parametrize("backend", ['numpy'])
    def test_1d_operator(self, backend):
        bm.set_backend(backend)
        domain = [0, 1]
        nx = 4
        extent = [0, nx]
        mesh = UniformMesh(domain, extent)
        convection_coef = bm.array([1.0], dtype=bm.float64)
        A0 = ConvectionOperator(mesh=mesh, convection_coef=convection_coef).assembly()
        A0_test = bm.to_numpy(A0.toarray())
        A1 = ConvectionOperator(mesh=mesh, convection_coef=convection_coef).assembly_central_const()
        A1_test = bm.to_numpy(A1.toarray())
        np.testing.assert_allclose(A0_test, A0_1d, atol=1e-1)
        np.testing.assert_allclose(A1_test, A1_1d, atol=1e-1)

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_2d_operator(self, backend):
        bm.set_backend(backend)
        domain = [0, 1, 0, 1]
        nx = 2
        ny = 2
        extent = [0, nx, 0, ny]
        mesh = UniformMesh(domain, extent)
        convection_coef = bm.array([1.0, -1.0], dtype=bm.float64)
        A0 = ConvectionOperator(mesh=mesh, convection_coef=convection_coef).assembly()
        A0_test = bm.to_numpy(A0.toarray())
        A1 = ConvectionOperator(mesh=mesh, convection_coef=convection_coef).assembly_central_const()
        A1_test = bm.to_numpy(A1.toarray())
        np.testing.assert_allclose(A0_test, A0_2d, atol=1e-10)
        np.testing.assert_allclose(A1_test, A1_2d, atol=1e-10)

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_3d_operator(self, backend):
        bm.set_backend(backend)
        domain = [0, 1, 0, 1, 0, 1]
        nx = 1
        ny = 1
        nz = 1
        extent = [0, nx, 0, ny, 0, nz]
        mesh = UniformMesh(domain, extent)
        convection_coef = bm.array([1.0, -1.0, -2.0], dtype=bm.float64)
        A0 = ConvectionOperator(mesh=mesh, convection_coef=convection_coef).assembly()
        A0_test = bm.to_numpy(A0.toarray())
        A1 = ConvectionOperator(mesh=mesh, convection_coef=convection_coef).assembly_central_const()
        A1_test = bm.to_numpy(A1.toarray())
        np.testing.assert_allclose(A0_test, A0_3d, atol=1e-10)
        np.testing.assert_allclose(A1_test, A1_3d, atol=1e-10)

if __name__ == "__main__":
    pytest.main(["-s", "-v", "test_convection_operator.py"])
