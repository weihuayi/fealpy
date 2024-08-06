# import numpy as bm
import pytest
from fealpy.experimental.backend import backend_manager as bm
# from fealpy.experimental.opt import CrayfishOptAlg 
# from fealpy.experimental.opt import HoneybadgerOptAlg 
from fealpy.experimental.opt import QuantumParticleSwarmOpt
# from fealpy.experimental.opt import SnowmeltOptAlg 
from fealpy.experimental.tests.opt.iopt_data import iopt_data
from fealpy.experimental.opt.optimizer_base import Optimizer, opt_alg_options

class TestIOptInterfaces:   
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_quantumparticleswarm_opt(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = lb + bm.random.rand(NP, data['ndim'])*(ub - lb)
        option = opt_alg_options( x0, data['objective'], data['domain'] , NP)
        optimizer = QuantumParticleSwarmOpt (option)
        gbest, gbest_f = optimizer.run()
        assert abs(gbest_f - data["optimal"]) < 100

    
 
if __name__ == "__main__":
    pytest.main(["./test_iopt_alg.py", "-k", "test_quantumparticleswarm_opt"])
    
