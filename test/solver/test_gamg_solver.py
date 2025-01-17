
import pytest
from fealpy.backend import backend_manager as bm
from fealpy.solver import GAMGSolver 

from gamg_solver_data import * 

class TestGAMGSolverInterfaces:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    @pytest.mark.parametrize("data", init_data)
    def test_init(self, data, backend):
        bm.set_backend(backend)
        solver = GAMGSolver(**data) 
        assert solver is not None
        assert solver.maxit == data['maxit']
        


if __name__ == "__main__":
    pytest.main(["./test_gamg_solver.py",'-k' ,"test_init"])
