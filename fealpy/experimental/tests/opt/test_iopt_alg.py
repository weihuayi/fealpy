import numpy as np
import pytest
from ...backend import backend_manager as bm
from ...opt import CrayfishOptAlg 
from ...opt import initialize
from ...opt.benchmark import iopt_benchmark_data

class TestIOptInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", iopt_benchmark_data)
    def test_crayfish_opt_alg(self, backend, data):
        X = bm.random.rand(self.N, self.dim) * (self.ub - self.lb) + self.lb
        pass
 
if __name__ == "__main__":
