import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.opt import CrayfishOptAlg 
from fealpy.experimental.tests.opt.iopt_data import iopt_data

class TestIOptInterfaces:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [10, 20, 30, 40])
    def test_crayfish_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = lb + bm.random.rand(NP, data['ndim'])*(ub - lb)
        problem = CrayfishOptAlg.get_options(x0, data['objective'], NP)
        optimizer = CrayfishOptAlg(problem)
        optimizer.run()

 
if __name__ == "__main__":
    pytest.main(["./test_iopt_alg.py", "-k", "test_crayfish_opt_alg"])
