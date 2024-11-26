import pytest

from fealpy.backend import backend_manager as bm
from fealpy.opt import CrayfishOptAlg 
from fealpy.opt import HoneybadgerAlg 
from fealpy.opt import QuantumParticleSwarmOpt, LevyQuantumParticleSwarmOpt
from fealpy.opt import SnowAblationOpt
from fealpy.opt import GreyWolfOptimizer
from fealpy.opt import HippopotamusOptAlg
from fealpy.opt import CrestedPorcupineOpt
from fealpy.opt import BlackwingedKiteAlg
from fealpy.opt import CuckooSearchOpt
from fealpy.opt import ParticleSwarmOpt
from fealpy.opt import ButterflyOptAlg
from fealpy.opt import ExponentialTrigonometricOptAlg
from fealpy.opt import DifferentialEvolution
from fealpy.opt import DifferentialtedCreativeSearch
from fealpy.opt import initialize
from fealpy.opt.optimizer_base import Optimizer, opt_alg_options

from iopt_data import iopt_data


class TestIOptInterfaces:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_crayfish_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = CrayfishOptAlg(option)
        optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_honeybadger_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = HoneybadgerAlg(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_quantumparticleswarm_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = QuantumParticleSwarmOpt(option)
        optimizer = LevyQuantumParticleSwarmOpt(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_snowmelt_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = SnowAblationOpt(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_hippopotamus_optimizer(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = HippopotamusOptAlg(option)
        optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_grey_wolf_optimizer(self, backend, data, NP):  
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = GreyWolfOptimizer(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [130])
    def test_crested_porcupine_opt(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = CrestedPorcupineOpt(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [130])
    def test_black_winged_kite_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = BlackwingedKiteAlg(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [130])
    def test_cuckoo_search_opt(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = CuckooSearchOpt(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [130])
    def test_particle_swarm_opt(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = ParticleSwarmOpt(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [130])
    def test_butterfly_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = ButterflyOptAlg(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [130])
    def test_exponential_trigonometric_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = ExponentialTrigonometricOptAlg(option)
        gbest, gbest_f = optimizer.run()
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [130])
    def test_differential_evolution(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = DifferentialEvolution(option)
        gbest, gbest_f = optimizer.run()

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [100])
    def test_differentialted_creative_search(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = DifferentialtedCreativeSearch(option)
        gbest, gbest_f = optimizer.run()

if __name__ == "__main__":
    # pytest.main(["./test_iopt_alg.py", "-k", "test_honeybadger_opt_alg"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_crayfish_opt_alg"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_quantumparticleswarm_opt_alg"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_grey_wolf_optimizer"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_snowmelt_opt_alg"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_hippopotamus_optimizer"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_crested_porcupine_opt"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_black_winged_kite_alg"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_cuckoo_search_opt"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_particle_swarm_opt"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_butterfly_opt_alg"])
    # pytest.main(["./test_iopt_alg.py", "-k", "test_differential_evolution"])    
    # pytest.main(["./test_iopt_alg.py", "-k", "test_exponential_trigonometric_opt_alg"])
    pytest.main(["./test_iopt_alg.py", "-k", "test_differentialted_creative_search"])    