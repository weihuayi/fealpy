import numpy as np
import pytest
import scipy.sparse as sp
import asyncio

from fealpy.backend import backend_manager as bm
from fealpy.solver import DirectSolverManager
from fealpy.sparse import COOTensor, CSRTensor

class TestDirectSolverManager:
    
    def _make_data(self, shape=(1000,1000), density=0.2, backend='numpy', device='cpu'):
        # 随机稀疏矩阵 A，保证非奇异

        bm.set_backend(backend)
        A = sp.rand(*shape, density=density)
        A = A + sp.eye(shape[0])
        A = A.tocoo()

        x_true = np.random.rand(shape[0])
        b_np   = A.dot(x_true)

        # 转为 FEALPy tensor
        A_t = COOTensor.from_scipy(A)
        b_t = bm.tensor(b_np)
        x_t = bm.tensor(x_true)

      
        if backend == 'pytorch':
            bm.set_default_device(device)
            if device != 'cpu':
                A_t = A_t.device_put(device)
        print(bm.backend_name)
        print(type(A_t.indices))

        return A_t, x_t, b_t

    @pytest.mark.parametrize("solver_name", ["scipy", "mumps", "pardiso", "cholmod"])
    @pytest.mark.parametrize("backend,device", [
        ("numpy", "cpu"),
        ("pytorch", "cpu"),
    ])
    def test_solve_cpu(self, solver_name, backend, device):
        A, x_true, b = self._make_data(backend=backend, device=device)

        mgr = DirectSolverManager()
        # 设定矩阵，默认 matrix_type='G'
        mgr.set_matrix(A, matrix_type='G')
        # 显式初始化 solver
    
        mgr.set_solver(solver_name=solver_name)
        # 同步求解
        x0 = mgr.solve(b)
        # 将解转换为 numpy
        x0_np = bm.tensor(x0)
        x_true_np = bm.to_numpy(x_true)
        # 验证解的准确性
        assert np.allclose(x0_np, x_true_np, atol=1e-8)

    def test_gpu_cupy(self):
        try:
            import cupy
        except ImportError:
            pytest.skip("Cupy not available")

        # 使用 pytorch->cuda 后端生成数据，再在 cupy 求解器上测试
        A, x_true, b = self._make_data(backend="numpy", device="cuda")

        mgr = DirectSolverManager()
        mgr.set_matrix(A, matrix_type='G')
        mgr.set_solver(solver_name="cupy")

        x0 = mgr.solve(b)

    def test_missing_set_matrix(self):
        """未调用 set_matrix 时，调用 solve 应报 ValueError"""
        mgr = DirectSolverManager(solver_name="scipy")
        with pytest.raises(ValueError):
            _ = mgr.solve(b=np.zeros(5))

    def test_invalid_solver(self):
        """指定不存在的 solver，应抛出 SolverNotAvailableError"""
        A, _, _ = self._make_data()
        mgr = DirectSolverManager(solver_name="no_such_solver")
        mgr.set_matrix(A, matrix_type='G')
        with pytest.raises(SolverNotAvailableError):
            mgr.set_solver()

if __name__ == "__main__":
    # 快速运行部分测试以观察性能
    import time
    test = TestDirectSolverManager()
    start = time.time()
    test.test_solve_cpu("scipy", "numpy", "cpu")
    print("scipy numpy cpu:", time.time() - start)
    start = time.time()
    test.test_solve_cpu("scipy", "pytorch", "cpu")
    print("scipy pytorch cpu:", time.time() - start)
    # start = time.time()
    # test.test_gpu_cupy()
    # print("cupy test:", time.time() - start)
    start = time.time()
    test.test_solve_cpu("mumps", "numpy", "cpu")
    print("mumps numpy cpu:", time.time() - start)
    start = time.time()
    test.test_solve_cpu("pardiso", "numpy", "cpu")
    print("pardiso numpy cpu:", time.time() - start)
