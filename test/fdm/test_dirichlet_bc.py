from fealpy.fdm import DirichletBC
from fealpy.backend import backend_manager as bm
import pytest
from dirichlet_bc_data import (pde, mesh, bd_1, A_before, A_after_none, A_after_bd1, 
                               f_before, f_after_none, f_after_bd1)

class TestDirichletBC:
    @pytest.mark.parametrize("backend", ['numpy'])
    @pytest.mark.parametrize("threshold, A_expected, f_expected", [
        (None, A_after_none, f_after_none),
        (bd_1, A_after_bd1, f_after_bd1),
    ])
    def test_apply_dirichlet_bc(self, backend, threshold, A_expected, f_expected):
        bm.set_backend(backend)
        A = A_before
        f = f_before.copy()
        uh = bm.zeros(f.shape[0])
        dbc = DirichletBC(mesh=mesh, gd=pde.dirichlet, threshold=threshold)
        A, f = dbc.apply(A, f, uh=uh)
        threshold_name = "None" if threshold is None else threshold.__name__
        assert bm.allclose(A.to_dense(), A_expected.to_scipy().toarray(), atol=1e-10), (
            f"A matrix mismatch for threshold={threshold_name}:\n"
            f"Expected:\n{A_expected.to_dense()}\nGot:\n{A.to_scipy().toarray()}"
        )
        assert bm.allclose(f, f_expected, atol=1e-10), (
            f"f vector mismatch for threshold={threshold_name}:\n"
            f"Expected:\n{f_expected}\nGot:\n{f}"
        )

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_sparse_matrix(self, backend):
        bm.set_backend(backend)
        A = A_before 
        f = f_before.copy()
        uh = bm.zeros(f.shape[0])
        dbc = DirichletBC(mesh=mesh, gd=pde.dirichlet, threshold=None)
        A_new, f_new = dbc.apply(A, f, uh=uh)
        assert bm.allclose(A_new.to_dense(), A_after_none.to_scipy().toarray(), atol=1e-10)
        assert bm.allclose(f_new, f_after_none, atol=1e-10)

    @pytest.mark.parametrize("backend", ['numpy'])
    def test_invalid_threshold(self, backend):
        bm.set_backend(backend)
        A = A_before
        f = f_before.copy()
        uh = bm.zeros(f.shape[0])
        def bad_threshold(node):
            return node[:, 0]
        dbc = DirichletBC(mesh=mesh, gd=pde.dirichlet, threshold=bad_threshold)
        with pytest.raises(IndexError):
            A_new, f_new = dbc.apply(A, f, uh=uh)

if __name__ == "__main__":
    pytest.main(["-s", "-v", "./test_dirichlet_bc.py"])