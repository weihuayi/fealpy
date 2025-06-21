import taichi as ti
import pytest
from fealpy.ti.sparse import CSRMatrix

ti.init(arch=ti.cuda)

def setup_csr_matrix():
    data = ti.field(dtype=ti.f64, shape=(5,))
    indices = ti.field(dtype=ti.i32, shape=(5,))
    indptr = ti.field(dtype=ti.i32, shape=(4,))
    
    data[0], data[1], data[2], data[3], data[4] = 1.0, 2.0, 3.0, 4.0, 5.0
    indices[0], indices[1], indices[2], indices[3], indices[4] = 0, 1, 0, 2, 1
    indptr[0], indptr[1], indptr[2], indptr[3] = 0, 2, 3, 5
    
    shape = (3, 3)
    csr_matrix = CSRMatrix((data, indices, indptr), shape)
    return csr_matrix

def test_csr_matrix_initialization(setup_csr_matrix):
    csr_matrix = setup_csr_matrix
    assert csr_matrix.data.shape[0] == 5
    assert csr_matrix.indices.shape[0] == 5
    assert csr_matrix.indptr.shape[0] == 4

def test_matvec(setup_csr_matrix):
    csr_matrix = setup_csr_matrix
    vec = ti.field(dtype=ti.f64, shape=(3,))
    result = ti.field(dtype=ti.f64, shape=(3,))
    
    vec[0], vec[1], vec[2] = 1.0, 2.0, 3.0
    
    expected_result = [5.0, 3.0, 22.0]

    result = csr_matrix@vec

    print(csr_matrix.tofield())
    print(vec)
    print(result)


    
    for i in range(3):
        assert result[i] == pytest.approx(expected_result[i], rel=1e-5)

if __name__ == "__main__":
    m = setup_csr_matrix()
    test_csr_matrix_initialization(m)
    test_matvec(m)

