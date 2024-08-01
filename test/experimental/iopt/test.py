import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.quadrangle_mesh import QuadrangleMesh
import test_iopt_alg

class Test_iopt:
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
    
