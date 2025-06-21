# test_csr_tensor.py
import pytest

from fealpy.sparse import csr_matrix as fpy_csr_matrix
from fealpy.sparse.csr_tensor import CSRTensor
from fealpy.backend import backend_manager as bm

ALL_BACKENDS = ['numpy', 'pytorch']


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_dims_and_shape(backend):
    bm.set_backend(backend)
    crow = bm.tensor([0,3,4,5])
    col = bm.tensor([1,2,2,0,0])
    values = bm.tensor([[1, 2, 5, 3, 4], [6, 7,10, 8, 9]], dtype=bm.float64) 
    sparse_shape = bm.tensor([3, 3])
    csr = CSRTensor(crow, col, values, sparse_shape)

    assert csr.ndim == 3
    assert csr.sparse_ndim == 2
    assert csr.dense_ndim == 1

    assert csr.shape == (2, 3, 3)
    assert csr.sparse_shape == (3, 3)
    assert csr.dense_shape == (2, )

@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_to_dense(backend):
    bm.set_backend(backend)
    crow = bm.tensor([0, 2, 3, 4])
    col = bm.tensor([1, 2, 0, 0])
    values = bm.tensor([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=bm.float64)
    sparse_shape = bm.tensor([3, 3])
    csr = CSRTensor(crow, col, values, sparse_shape)

    arr = csr.to_dense()
    assert arr.dtype == bm.float64
    bm.allclose(
        arr,
        bm.tensor([[[0, 1, 2],
                    [3, 0, 0],
                    [4, 0, 0]],
                   [[0, 6, 7],
                    [8, 0, 0],
                    [9, 0, 0]]], dtype=bm.float64)
    )
    csr2 = CSRTensor(crow, col, None, sparse_shape)
    arr2 = csr2.to_dense(fill_value=1.22)
    bm.allclose(
        arr2,
        bm.tensor([[[0, 1.22, 1.22],
                    [1.22, 0, 0],
                    [1.22, 0, 0]],
                   [[0, 1.22, 1.22],
                    [1.22, 0, 0],
                    [1.22, 0, 0]]], dtype=bm.float64)
    )

@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_tril(backend):
    bm.set_backend(backend)
    crow = bm.tensor([0, 0, 2, 3, 4, 4])
    col = bm.tensor([1, 2, 3, 2])
    values = bm.tensor([4, 3, 1, 2], dtype=bm.float32)
    spshape = (5, 4)
    csr_tensor = CSRTensor(crow, col, values, spshape)
    tril_tensor = csr_tensor.tril(k=0)

    expected_crow = bm.tensor([0, 0, 1, 1, 2, 2])
    expected_col = bm.tensor([1, 2])
    expected_values = bm.tensor([4, 2], dtype=bm.float32)

    assert bm.all(bm.equal(tril_tensor.crow, expected_crow))
    assert bm.all(bm.equal(tril_tensor.col, expected_col))
    assert bm.allclose(tril_tensor.values, expected_values)

def create_csr_tensor(crow, col, values, shape):
    return CSRTensor(crow=crow, col=col, values=values, spshape=shape)

# CSRTensor.add 测试用例
class TestCSRTensorAdd:
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_add_csr_tensor(self, backend):
        bm.set_backend(backend)
        # 初始化两个 CSRTensors
        csr1 = create_csr_tensor(
            crow=bm.tensor([0, 1, 1, 2, 2]), col=bm.tensor([1, 3]), values=bm.tensor([1, 2]),
            shape=(4, 4)
        )
        csr2 = create_csr_tensor(
            crow=bm.tensor([0, 1, 2, 2, 2]), col=bm.tensor([2, 3]), values=bm.tensor([3, 4]),
            shape=(4, 4)
        )
        csr3 = create_csr_tensor(
            crow=bm.tensor([0, 1, 2, 2, 2]), col=bm.tensor([2, 3]), values=None,
            shape=(4, 4)
        )
        csr4 = create_csr_tensor(
            crow=bm.tensor([0, 1, 1, 2, 2]), col=bm.tensor([1, 3]), values=None,
            shape=(4, 4)
        )

        # 执行 add 操作
        result1 = csr1.add(csr2, alpha=2)
        result2 = csr3.add(csr4, alpha=2)
        result3 = csr1.add(csr3, alpha=2) 

        # 验证结果
        expected_crow1 = bm.tensor([0, 2, 3, 4, 4])
        expected_col1 = bm.tensor([1, 2, 3, 3])
        expected_values1 = bm.tensor([1, 6, 8, 2])

        assert bm.allclose(result1._crow, expected_crow1)
        assert bm.allclose(result1._col, expected_col1)
        assert bm.allclose(result1._values, expected_values1)

        expected_csr2 = create_csr_tensor(
            crow=bm.tensor([0, 2, 3, 4, 4]), col=bm.tensor([1, 2, 3, 3]), values=bm.tensor([2., 1., 1., 2.]),
            shape=(4, 4)
        )
        expected_csr3 = create_csr_tensor(
            crow=bm.tensor([0, 2, 3, 4, 4]), col=bm.tensor([1, 2, 3, 3]), values=bm.tensor([1., 2., 2., 2.]),
            shape=(4, 4)
        )
        assert bm.allclose(expected_csr2.toarray(), result2.toarray())
        assert bm.allclose(expected_csr3.toarray(), result3.toarray())

    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_add_tensor(self, backend):
        bm.set_backend(backend)
        # 初始化一个 CSRTensor 和一个 dense Tensor
        csr = create_csr_tensor(crow=bm.tensor([0,1,1,1,1]),col=bm.tensor([2]),
                                values=bm.tensor([[1]], dtype=bm.float64),
                                shape=(4, 4))
        tensor = bm.zeros((1, 4, 4), dtype=bm.float64)

        # 执行 add 操作
        result = csr.add(tensor)

        # 验证结果
        expected_tensor = bm.tensor([[[0., 0., 1., 0.],
                                         [0., 0., 0., 0.],
                                         [0., 0., 0., 0.],
                                         [0., 0., 0., 0.]]], dtype=bm.float64)
        assert bm.allclose(result, expected_tensor)

    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_add_number(self, backend):
        bm.set_backend(backend)
        # 初始化一个 COOTensor 和一个数值
        csr = create_csr_tensor(crow=bm.tensor([0,1,1,1,1]),col=bm.tensor([2]), values=bm.tensor([[1], [2]]), shape=(4, 4))
        number = 2

        # 执行 add 操作
        result = csr.add(number)

        # 验证结果的值（注意，这里只是演示，实际上 result 仍然是 COOTensor 类型）
        assert bm.allclose(result._values, bm.tensor([[3], [4]]))


class TestCSRTensorMatmul():
    @pytest.mark.parametrize("backend", ALL_BACKENDS)
    def test_matmul_sparse(self, backend):
        bm.set_backend(backend)
        m1 = bm.array([
            [1, 0, 0, 3],
            [0, 1, 2, 4],
            [0, 0, 1, 0],
            [3, 0, 5, 1]
        ], dtype=bm.float64)
        m2 = bm.array([
            [6, 7, 0, 0],
            [3, 4, 0, 0],
            [2, 0, 5, 0],
            [0, 0, 0, 1]
        ], dtype=bm.float64)
        m3 = m1 @ m2
        csr1 = fpy_csr_matrix(m1)
        csr2 = fpy_csr_matrix(m2)
        csr3 = csr1 @ csr2
        assert bm.allclose(csr3.toarray(), m3)
        