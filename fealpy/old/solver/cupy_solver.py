import cupy as cp
import numpy as np
from scipy.sparse import issparse
import cupyx.scipy.sparse.linalg as cpx
import time

from scipy.sparse import issparse
from cupyx.scipy.sparse.linalg import LinearOperator as CuPyLinearOperator

class CupySolver():

    def __init__(self):
        pass

    def cg_solver(self, A, b, atol=1e-18):
        # 将数组及矩阵从 numpy 转化为 cupy
        A_gpu = self.np_matrix_to_cp(A)
        b_gpu = self.np_array_to_cp(b)

        # 在GPU上求解，并计时
        start_time_gpu = time.time()
        x_gpu, info = cpx.cg(A_gpu, b_gpu, atol=atol)
        end_time_gpu = time.time()

        # 输出GPU求解时间
        gpu_time = end_time_gpu - start_time_gpu
        print("cupy cg GPU time: {:.5f} seconds".format(gpu_time))
        return x_gpu.get()

    def gmres_solver(self, A, b, atol=1e-18):
        # 将数组及矩阵从 numpy 转化为 cupy
        A_gpu = self.np_matrix_to_cp(A)
        b_gpu = self.np_array_to_cp(b)

        # 在GPU上求解，并计时
        start_time_gpu = time.time()
        x_gpu, info = cpx.gmres(A_gpu, b_gpu, atol=atol)
        end_time_gpu = time.time()

        # 输出GPU求解时间
        gpu_time = end_time_gpu - start_time_gpu
        print("cupy gmres GPU time: {:.5f} seconds".format(gpu_time))
        return x_gpu.get()

    def np_matrix_to_cp(self, A):
        if type(A) is np.array:
            A_gpu = cp.array(A, dtype=cp.float64)
        elif issparse(A):
            A_gpu = cp.sparse.csr_matrix(A.astype(cp.float64))
        else:
            print("The format of A cannot be converted at the moment, please add")
        return A_gpu

    def np_array_to_cp(self, b):
        b_gpu = cp.array(b, dtype=cp.float64)
        return b_gpu


