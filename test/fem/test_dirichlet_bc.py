
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import DirichletBC
from fealpy.sparse import coo_matrix, COOTensor, CSRTensor


@pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
def test_apply_csr_matrix(backend):
    bm.set_backend(backend)
    mesh = TriangleMesh.from_box([-1, 1, -1, 1], nx=2, ny=2)
    space = LagrangeFESpace(mesh, p=3)
    gdof = space.number_of_global_dofs()
    A = bm.zeros((gdof * gdof, ), dtype=space.ftype, device=space.device)
    flag = bm.random.rand(gdof*gdof) < 0.4
    src = bm.random.rand(gdof*gdof)[flag]
    A = bm.set_at(A, flag, bm.astype(src, space.ftype))
    A = A.reshape(gdof, gdof)
    A_COO = coo_matrix(A)
    A_CSR = A_COO.tocsr()

    dbc = DirichletBC(space)
    coo_result = dbc.apply_matrix(A_COO)
    csr_result = dbc.apply_matrix(A_CSR)

    assert isinstance(coo_result, COOTensor)
    assert isinstance(csr_result, CSRTensor)
    assert bm.allclose(A_COO.toarray(), A_CSR.toarray())
