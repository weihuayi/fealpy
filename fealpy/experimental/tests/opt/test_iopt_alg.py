import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.opt import CrayfishOptAlg 
from fealpy.experimental.opt import HoneybadgerOptAlg 
from fealpy.experimental.opt import QuantumParticleswarmOptAlg 
from fealpy.experimental.opt import SnowmeltOptAlg 
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

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [10, 20, 30, 40])
    def test_honeybadger_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = lb + bm.random.rand(NP, data['ndim'])*(ub - lb)
        problem = HoneybadgerOptAlg.get_options(x0, data['objective'], NP)
        optimizer = HoneybadgerOptAlg(problem)
        optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [10, 20, 30, 40])
    def test_quantumparticleswarm_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = lb + bm.random.rand(NP, data['ndim'])*(ub - lb)
        problem = QuantumParticleswarmOptAlg.get_options(x0, data['objective'], NP)
        optimizer = QuantumParticleswarmOptAlg(problem)
        optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [10, 20, 30, 40])
    def test_snowmelt_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = lb + bm.random.rand(NP, data['ndim'])*(ub - lb)
        problem = SnowmeltOptAlg.get_options(x0, data['objective'], NP)
        optimizer = SnowmeltOptAlg(problem)
        optimizer.run()
 
if __name__ == "__main__":
    #pytest.main(["./test_iopt_alg.py", "-k", "test_crayfish_opt_alg"])
    #pytest.main(["./test_iopt_alg.py", "-k", "test_honeybadger_opt_alg"])
    #pytest.main(["./test_iopt_alg.py", "-k", "test_quantumparticleswarm_opt_alg"])
    pytest.main(["./test_iopt_alg.py", "-k", "test_snowmelt_opt_alg"])
    
