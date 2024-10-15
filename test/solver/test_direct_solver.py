
import numpy as np
import pytest
import scipy.sparse as sp

from fealpy.backend import backend_manager as bm
from fealpy.solver import spsolve
from fealpy.sparse import COOTensor, CSRTensor

class TestDirectSolver:

    def _check_solution(self, x0, x, tolerance=1e-5):
        residual = bm.max(bm.abs(x0-x))
        return residual < tolerance

    def _get_cpu_data(self):
        A = sp.rand(10, 10, density=0.3)  # 随机生成一个稀疏矩阵
        A = A + sp.eye(10)  # 保证矩阵非奇异
        A = A.tocoo()
        x = np.random.rand(10)
        b = A.dot(x)

        A = COOTensor.from_scipy(A)
        b = bm.tensor(b)
        x = bm.tensor(x)
        return A, x, b

    def _get_gpu_data(self):
        A, x, b = self._get_cpu_data()
        A = A.device_put('cuda')
        print(A.indices().device, A.values().device, b.device, x.device)
        return A, x, b

    @pytest.mark.parametrize('backend', ['numpy', 'pytorch'])
    @pytest.mark.parametrize('solver_type', ['scipy', 'mumps', 'cupy'])
    def test_cpu(self, backend, solver_type):
        bm.set_backend(backend)
        solver = lambda A, b: spsolve(A, b, solver_type)
        A, x, b = self._get_cpu_data()
        x0 = solver(A, b) 
        assert self._check_solution(x0, x), "f{backend} Test failed!!!!!!!!!!!!!!!!!!!!!!!!"

    def test_gpu(self):
        bm.set_backend("pytorch")
        bm.set_default_device("cuda")
        solver = lambda A, b: spsolve(A, b, 'cupy')

        A, x, b = self._get_gpu_data()
        x0 = solver(A, b)
        assert self._check_solution(x0, x), "Pytorch GPU test failed!!!!!!!!!!!!!!!!!!!!!!!!"
        print("Pytorch GPU test passed!")

if __name__ == '__main__':
    test = TestDirectSolver()
    #test.test_cpu('numpy', 'scipy')
    #test.test_cpu('numpy', 'mumps')
    #test.test_cpu('numpy', 'cupy')
    #test.test_gpu()

    #test1 = TestDirectSolver(solver_type='mumps')
    #test1.test_cpu()

    pytest.main(['test_cpu', "-q"])   
    pytest.main(['test_gpu', "-q"])   



