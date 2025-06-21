
import pytest
import torch
from fealpy.torch.sparse._spspmm import spspmm_coo
from fealpy.torch.sparse import COOTensor


def test_spspmm_coo_shapes_mismatch():
    indices1 = torch.tensor([[0, 0], [1, 1]])
    values1 = torch.tensor([1.0, 2.0])
    spshape1 = torch.Size([2, 3])

    indices2 = torch.tensor([[0, 1], [0, 1]])
    values2 = torch.tensor([1.0, 1.0])
    spshape2 = torch.Size([2, 3])

    with pytest.raises(ValueError):
        spspmm_coo(indices1, values1, spshape1, indices2, values2, spshape2)


def test_spspmm_coo_valid_input():
    indices1 = torch.tensor([[0, 0, 1, 2],
                             [0, 1, 1, 2]])
    values1 = torch.tensor([1., 3., 4., 2.], dtype=torch.float64)
    spshape1 = (3, 3)

    indices2 = torch.tensor([[0, 1, 2],
                             [1, 0, 0]])
    values2 = torch.tensor([2., 9., 3.], dtype=torch.float64)
    spshape2 = (3, 2)

    indices, values, output_shape = spspmm_coo(indices1, values1, spshape1, indices2, values2, spshape2)
    sparse_coo = COOTensor(indices, values, output_shape)
    sparse_coo = sparse_coo.coalesce()
    result = sparse_coo.to_dense()

    expected = torch.tensor([[27., 2.],
                             [36., 0.],
                             [6., 0.]], dtype=torch.float64)

    assert torch.allclose(result, expected)

# Additional tests can be added here to cover more edge cases, different shapes,
# or to ensure consistency with other matrix multiplication methods under various conditions.
