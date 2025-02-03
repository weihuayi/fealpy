import pytest

from fealpy.backend import backend_manager as bm
from fealpy.opt import *
from fealpy.opt.optimizer_base import Optimizer, opt_alg_options

from iopt_data import iopt_data


class TestIOptInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", iopt_data)
    @pytest.mark.parametrize("NP", [130])
    def test_opt_alg(self, backend, data, NP):
        bm.set_backend(backend)
        lb, ub = data['domain']
        x0 = initialize(NP, data['ndim'], ub, lb)
        option = opt_alg_options(x0, data['objective'], data['domain'], NP)
        optimizer = TeachingLearningBasedAlg(option)
        # optimizer = SineCosineAlg(option)
        # optimizer = MossGrowthOpt(option)
        # optimizer = RimeOptAlg(option)
        # optimizer = MarinePredatorsAlg(option)
        # optimizer = HoneybadgerAlg(option)
        # optimizer = CrayfishOptAlg(option)
        # optimizer = QuantumParticleSwarmOpt(option)
        # optimizer = LevyQuantumParticleSwarmOpt(option)
        # optimizer = SnowAblationOpt(option)
        # optimizer = HippopotamusOptAlg(option)
        # optimizer = GreyWolfOptimizer(option)
        # optimizer = CrestedPorcupineOpt(option)
        # optimizer = BlackwingedKiteAlg(option)
        # optimizer = CuckooSearchOpt(option)
        # optimizer = ParticleSwarmOpt(option)
        # optimizer = ButterflyOptAlg(option)
        # optimizer = ExponentialTrigonometricOptAlg(option)
        # optimizer = DifferentialEvolution(option)
        # optimizer = DifferentialtedCreativeSearch(option)
        # optimizer = CuckooQuantumParticleSwarmOpt(option)
        optimizer.run()

if __name__ == "__main__":
    pytest.main(["./test_iopt_alg.py", "-k", "test_opt_alg"])
      