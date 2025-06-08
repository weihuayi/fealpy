from fealpy.backend import backend_manager as bm
from fealpy.fdm import DirichletBC
from fealpy.sparse import CSRTensor
from fealpy.mesh import UniformMesh
import pytest
from dirichlet_bc_data import (pde, domain, extent, bd_1, A_before, A_after_none, A_after_bd1, 
                               f_before, f_after_none, f_after_bd1)

class TestDirichletBC:
    @pytest.mark.parametrize("backend", ['numpy','pytorch','jax'])
    @pytest.mark.parametrize("threshold, A_expected, f_expected", [
        (None, A_after_none, f_after_none),
        (bd_1, A_after_bd1, f_after_bd1),
    ])
    def test_apply_dirichlet_bc(self, backend, threshold, A_expected, f_expected):
        bm.set_backend(backend)
        indptr_before = bm.array(A_before.indptr, dtype=bm.int64)
        indices_before = bm.array(A_before.indices, dtype=bm.int64)
        values_before = bm.array(A_before.values, dtype=bm.float64)
        A = CSRTensor(indptr_before, indices_before, values_before, A_before.shape)

        indptr_expected = bm.array(A_expected.indptr, dtype=bm.int64)
        indices_expected = bm.array(A_expected.indices, dtype=bm.int64)
        values_expected = bm.array(A_expected.values, dtype=bm.float64)
        A_expected = CSRTensor(indptr_expected, indices_expected, values_expected, A_expected.shape)

        f = bm.array(f_before.copy(), dtype=bm.float64)
        uh = bm.zeros(f.shape[0], dtype=bm.float64)
        mesh = UniformMesh(domain, extent)
        dbc = DirichletBC(mesh=mesh, gd=pde.dirichlet, threshold=threshold)
        A, f = dbc.apply(A, f, uh=uh)
        threshold_name = "None" if threshold is None else threshold.__name__
        A_dense = A.to_dense()
        A_expected_dense = A_expected.to_dense()
        assert bm.allclose(A_dense, A_expected_dense, atol=1e-10), (
            f"A matrix mismatch for threshold={threshold_name}:\n"
            f"Expected:\n{A_expected_dense}\nGot:\n{A_dense}"
        )
        assert bm.allclose(f, bm.array(f_expected), atol=1e-10), (
            f"f vector mismatch for threshold={threshold_name}:\n"
            f"Expected:\n{f_expected}\nGot:\n{f}"
        )
    
if __name__ == "__main__":
    pytest.main(["-s", "-v", "./test_dirichlet_bc.py"])