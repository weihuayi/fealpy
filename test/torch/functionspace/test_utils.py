
import torch

from fealpy.torch.functionspace.utils import (
    flatten_indices,
    to_tensor_dof,
    tensor_basis,
    normal_strain,
    shear_strain
)


def test_flatten_indices():
    shape = (2, 3, 4)
    permute = (0, 2, 1)
    indices = flatten_indices(shape, permute)

    expected = torch.tensor(
        [[[0, 3, 6, 9],
          [1, 4, 7, 10],
          [2, 5, 8, 11]],
         [[12, 15, 18, 21],
          [13, 16, 19, 22],
          [14, 17, 20, 23]]]
    )

    assert torch.equal(indices, expected)


def test_to_tensor_dof():
    to_dof = torch.tensor([[0, 1, 2],
                           [2, 3, 4],
                           [4, 5, 6],
                           [6, 7, 8]])
    dof_numel = 3
    gdof = 9
    result = to_tensor_dof(to_dof, dof_numel, gdof, dof_priority=False)
    expected = torch.tensor(
        [[0, 1, 2, 3, 4, 5, 6, 7, 8],
         [6, 7, 8, 9, 10, 11, 12, 13, 14],
         [12, 13, 14, 15, 16, 17, 18, 19, 20],
         [18, 19, 20, 21, 22, 23, 24, 25, 26]]
    )

    assert torch.equal(result, expected)

    result = to_tensor_dof(to_dof, dof_numel, gdof, dof_priority=True)
    expected = torch.tensor(
        [[0, 9, 18, 1, 10, 19, 2, 11, 10],
         [2, 11, 20, 3, 12, 21, 4, 13, 22],
         [4, 13, 22, 5, 14, 23, 6, 15, 24],
         [6, 15, 24, 7, 16, 25, 8, 27, 26]]
    )


def test_tensor_basis():
    result = tensor_basis((2, 3), dtype=torch.float32)
    expected = torch.tensor(
        [[[1, 0, 0],
          [0, 0, 0]],
         [[0, 1, 0],
          [0, 0, 0]],
         [[0, 0, 1],
          [0, 0, 0]],
         [[0, 0, 0],
          [1, 0, 0]],
         [[0, 0, 0],
          [0, 1, 0]],
         [[0, 0, 0],
          [0, 0, 1]]],
          dtype=torch.float32
    ) # (6, 2, 3)

    assert torch.allclose(result, expected)

    result = tensor_basis((3, ), dtype=torch.float32)
    expected = torch.tensor(
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
          dtype=torch.float32
    ) # (6, 3)

    assert torch.allclose(result, expected)


def test_normal_strain():
    gphi = torch.linspace(0.1, 1.2, 12).reshape(2, 2, 3)
    indices = flatten_indices((2, 3), (1, 0))
    ns = normal_strain(gphi, indices)
    expected = torch.tensor(
        [[[0.1, 0.4, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.2, 0.5, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.3, 0.6]],
         [[0.7, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.8, 1.1, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.9, 1.2]]],
    )
    assert torch.allclose(ns, expected)


def test_shear_strain():
    gphi = torch.linspace(0.1, 1.2, 12).reshape(2, 2, 3)
    indices = flatten_indices((2, 3), (1, 0))
    ss = shear_strain(gphi, indices)
    expected = torch.tensor(
        [[[0.2, 0.5, 0.1, 0.4, 0.0, 0.0],
          [0.3, 0.6, 0.0, 0.0, 0.1, 0.4],
          [0.0, 0.0, 0.3, 0.6, 0.2, 0.5]],
         [[0.8, 1.1, 0.7, 1.0, 0.0, 0.0],
          [0.9, 1.2, 0.0, 0.0, 0.7, 1.0],
          [0.0, 0.0, 0.9, 1.2, 0.8, 1.1]]],
    )
    assert torch.allclose(ss, expected)
