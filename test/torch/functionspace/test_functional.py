
import torch

from fealpy.torch.functionspace.functional import (
    generate_tensor_basis,
    generate_tensor_grad_basis
)


def test_generate_tensor_basis():
    basis = torch.tensor([[1, 2, 3], ], dtype=torch.float64)
    result = generate_tensor_basis(basis, (2, 2), dof_priority=False)
    expected = torch.tensor(
        [[
            [[1, 0],
             [0, 0]],
            [[0, 1],
             [0, 0]],
            [[0, 0],
             [1, 0]],
            [[0, 0],
             [0, 1]],
            [[2, 0],
             [0, 0]],
            [[0, 2],
             [0, 0]],
            [[0, 0],
             [2, 0]],
            [[0, 0],
             [0, 2]],
            [[3, 0],
             [0, 0]],
            [[0, 3],
             [0, 0]],
            [[0, 0],
             [3, 0]],
            [[0, 0],
             [0, 3]]
        ]],
        dtype=torch.float64
    )

    basis = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float64)
    result = generate_tensor_basis(basis, (2, 2), dof_priority=True)
    expected = torch.tensor(
        [[
            [[1, 0], [0, 0]],
            [[2, 0], [0, 0]],
            [[3, 0], [0, 0]],
            [[0, 1], [0, 0]],
            [[0, 2], [0, 0]],
            [[0, 3], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 0], [2, 0]],
            [[0, 0], [3, 0]],
            [[0, 0], [0, 1]],
            [[0, 0], [0, 2]],
            [[0, 0], [0, 3]]
        ],[
            [[4, 0], [0, 0]],
            [[5, 0], [0, 0]],
            [[6, 0], [0, 0]],
            [[0, 4], [0, 0]],
            [[0, 5], [0, 0]],
            [[0, 6], [0, 0]],
            [[0, 0], [4, 0]],
            [[0, 0], [5, 0]],
            [[0, 0], [6, 0]],
            [[0, 0], [0, 4]],
            [[0, 0], [0, 5]],
            [[0, 0], [0, 6]]
        ]],
        dtype=torch.float64
    )

    assert torch.allclose(result, expected)


def test_generate_tensor_grad_basis():
    basis = torch.tensor([[[1, 11], [2, 22], [3, 33]], ], dtype=torch.float64)
    result = generate_tensor_grad_basis(basis, (2, 2), dof_priority=False)
    expected = torch.tensor([[
            [[[1, 11], [0, 0]],
             [[0, 0], [0, 0]]],
            [[[0, 0], [1, 11]],
             [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]],
             [[1, 11], [0, 0]]],
            [[[0, 0], [0, 0]],
             [[0, 0], [1, 11]]],
            [[[2, 22], [0, 0]],
             [[0, 0], [0, 0]]],
            [[[0, 0], [2, 22]],
             [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]],
             [[2, 22], [0, 0]]],
            [[[0, 0], [0, 0]],
             [[0, 0], [2, 22]]],
            [[[3, 33], [0, 0]],
             [[0, 0], [0, 0]]],
            [[[0, 0], [3, 33]],
             [[0, 0], [0, 0]]],
            [[[0, 0], [0, 0]],
             [[3, 33], [0, 0]]],
            [[[0, 0], [0, 0]],
             [[0, 0], [3, 33]]]
        ]],
        dtype=torch.float64
    )

    assert torch.allclose(result, expected)
