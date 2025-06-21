
import pytest

from fealpy.backend import backend_manager as bm
import test_iopt_alg

class TestIOptInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("alg_name", ['COA', 'HBA', 'QPSO', 'SAO'])
    def test_init(self, backend, alg_name):
        bm.set_backend(backend)
        test_iopt_alg.test_alg(alg_name)
 
if __name__ == "__main__":
    alg_name = "SAO"
    backend = "pytorch"
    T = Test_iopt()
    T.test_init(backend, alg_name)
    
