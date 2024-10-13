import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.opt import CrayfishOptAlg 
from fealpy.experimental.opt import HoneybadgerOptAlg 
from fealpy.experimental.opt import QuantumParticleSwarmOptAlg 
from fealpy.experimental.opt import SnowmeltOptAlg 
from fealpy.experimental.opt import GreyWolfOptimizer
from fealpy.experimental.opt import HippopotamusOptAlg
from fealpy.experimental.tests.opt.iopt_data import iopt_data
from fealpy.experimental.opt.optimizer_base import Optimizer, opt_alg_options

class TestIOptInterfaces:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_crayfish_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = lb + bm.random.rand(NP, data['ndim'])*(ub - lb)
        option = opt_alg_options( x0, data['objective'], data['domain'] , NP)
        optimizer = CrayfishOptAlg(option)
        optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_honeybadger_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = lb + bm.random.rand(NP, data['ndim'])*(ub - lb)
        option = opt_alg_options( x0, data['objective'], data['domain'] , NP)
        optimizer = HoneybadgerOptAlg(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_quantumparticleswarm_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = lb + bm.random.rand(NP, data['ndim'])*(ub - lb)
        option = opt_alg_options( x0, data['objective'], data['domain'] , NP)
        optimizer = QuantumParticleSwarmOptAlg(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_snowmelt_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = lb + bm.random.rand(NP, data['ndim'])*(ub - lb)
        option = opt_alg_options( x0, data['objective'], data['domain'] , NP)
        optimizer = SnowmeltOptAlg(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_hippopotamus_optimizer(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = lb + bm.random.rand(NP, data['ndim'])*(ub - lb)
        option = opt_alg_options( x0, data['objective'], data['domain'] , NP)
        optimizer = HippopotamusOptAlg(option)
        optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_grey_wolf_optimizer(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = lb + bm.random.rand(NP, data['ndim'])*(ub - lb)
        option = opt_alg_options( x0, data['objective'], data['domain'] , NP)
        optimizer = GreyWolfOptimizer(option)
        gbest, gbest_f = optimizer.run()
    
if __name__ == "__main__":
    # pytest.main(["./test_iopt_alg.py", "-k", "test_honeybadger_opt_alg"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_crayfish_opt_alg"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_quantumparticleswarm_opt_alg"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_grey_wolf_optimizer"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_snowmelt_opt_alg"])
    pytest.main(["./test_iopt_alg.py", "-k", "test_hippopotamus_optimizer"])
    
