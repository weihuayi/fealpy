
import torch

from fealpy.torch.functionspace.utils import (
    flatten_indices,
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
