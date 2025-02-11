import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fealpy.backend import backend_manager as bm
from fealpy.solver import cg
from fealpy.sparse import COOTensor, CSRTensor
import time 

def SGS_preconditioner(A):
    """ 生成显式的 SGS 预处理矩阵 M^{-1} """
    A = A.tocsr()
    D = sp.diags(A.diagonal())  # 对角线元素
    D_inv = sp.diags(1 / A.diagonal())  # 计算 D^{-1}
    L = sp.tril(A, k=-1)  # 下三角部分
    U = sp.triu(A, k=1)   # 上三角部分

    # 计算 SGS 预处理矩阵 M_inv = (L + D)^(-1) * D * (D + U)^(-1)
    LD_inv = spla.inv(L + D)  # (L + D)^{-1}
    DU_inv = spla.inv(D + U)  # (D + U)^{-1}

    M_inv = LD_inv @ D_inv @ DU_inv  # 计算 M^{-1}
    M_inv = M_inv.tocsr()  # 转换为 CSR 格式
    return M_inv  # 返回稀疏矩阵格式

class TestPcgSolver:

    def _check_solution(self, x0, x, tolerance=1e-5):
        residual = bm.max(bm.abs(x0-x))
        return residual < tolerance

    def _get_data(self):
        n = 2000  # 矩阵大小
        # 生成标准的拉普拉斯矩阵
        diagonals = [-np.ones(n-1), 2*np.ones(n), -np.ones(n-1)]
        A = sp.diags(diagonals, [-1, 0, 1], shape=(n, n), format="csr")
        
        # 使用拉普拉斯矩阵生成预处理矩阵 B
        B = SGS_preconditioner(A) # 使用 A 的对角线倒数作为预处理矩阵
        x = np.random.rand(n)
        b = A @ x
        # 转换为 Fealpy 的 CSRTensor 格式
        A = CSRTensor.from_scipy(A)
        B = CSRTensor.from_scipy(B)
        b = bm.tensor(b)
        x = bm.tensor(x)

        return A, x, b, B

    @pytest.mark.parametrize('backend', ['numpy', 'pytorch'])
    def test_cg(self, backend):
        bm.set_backend(backend)
        A, x, b,B = self._get_data()
        start_time  = time.time()
        x0 = cg(A, b)
        end_time = time.time()
        print("Time cost: ",end_time-start_time)
        assert self._check_solution(x0, x), "f{backend} Test failed!!!!!!!!!!!!!!!!!!!!!!!!"


if __name__ == '__main__':
    test = TestPcgSolver() 
    test.test_cg('numpy')