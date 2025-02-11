import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fealpy.backend import backend_manager as bm
from fealpy.solver import GaussiSeidei 
from fealpy.sparse import COOTensor, CSRTensor
from gamg_solver_data import * 
import time 

class TestGSSolver:
    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", test_data)
    def test_gs(self, backend):
        bm.set_backend(backend)
        A = test_data[0]['A']
        f = test_data[0]['f']
        x0 = test_data[0]['sol']

        start_time  = time.time()
        phi,res,iter = GaussiSeidei.gs(A,f)
        end_time = time.time()
        err = bm.sqrt(bm.sum((phi-x0)**2))/bm.sqrt(bm.sum(x0**2))

        res_0 = bm.array(f)
        res_0 = bm.sqrt(bm.sum(res_0**2))
        stop_res = res/res_0
        # 输出误差和残差
        print('err:', err)
        print('stop_res:',res )

        # 判断收敛
        rtol = 1e-8  # 设置收敛阈值
        if stop_res <= rtol:
            print("Converged: True")
            converged = True
        else:
            print("Converged: False")
            converged = False

        # 断言确保收敛
        assert converged, f"GS solver did not converge: stop_res = {stop_res} > rtol = {rtol}"

        print("Time cost: ",end_time-start_time)


if __name__ == '__main__':
    test = TestGSSolver() 
    test.test_gs('numpy')