
import pytest
import torch
from fealpy.torch.sparse._spmm import spmm_coo


def test_spmm_coo_1d_vector():
    # Define test inputs
    indices = torch.tensor([[0, 0, 0, 1, 2, 2, 3, 3],
                            [0, 2, 3, 2, 0, 3, 1, 3]])
    values = torch.tensor([[1, 2, 4, -1, 3, -1, 5, -2],
                           [-1, -2, -4, 1, -3, 1, -5, 2]], dtype=torch.float32)
    spshape = (4, 4)
    x = torch.tensor([-3, -1, 1, 2], dtype=torch.float32)

    # Expected output
    expected = torch.tensor([[7, -1, -11, -9],
                             [-7, 1, 11, 9]], dtype=torch.float32)

    # Perform the test
    output = spmm_coo(indices, values, spshape, x)

    # Check if the output is as expected
    assert torch.allclose(output, expected), f"Expected {expected} but got {output}"


def test_batched_spmm_coo_2d_vector():
    # Define test inputs
    indices = torch.tensor([[0, 0, 0, 1, 2, 2, 2],
                            [0, 2, 3, 2, 0, 1, 3]])
    values = torch.tensor([[1, 2, 4, -1, 3, 2, 5],
                           [-1, -2, -4, 1, -3, -2, -5]], dtype=torch.float32)
    spshape = (3, 4)
    x = torch.tensor([[-1, -1, -1, -1, -1],
                      [6, 9, 1, 2, 7],
                      [2, 2, 2, 2, 1],
                      [1, 8, 2, 2, 5]], dtype=torch.float32)

    # Expected output
    expected = torch.tensor([[[7, 35, 11, 11, 21],
                              [-2, -2, -2, -2, -1],
                              [14, 55, 9, 11, 36]],
                             [[-7, -35, -11, -11, -21],
                              [2, 2, 2, 2, 1],
                              [-14, -55, -9, -11, -36]]], dtype=torch.float32)

    # Perform the test
    output = spmm_coo(indices, values, spshape, x)

    # Check if the output is as expected
    assert torch.allclose(output, expected), f"Expected {expected} but got {output}"


def test_spmm_coo_2d_batch_vector():
    # Define test inputs
    indices = torch.tensor([[0, 0, 0, 1, 2, 2, 2],
                            [0, 2, 3, 2, 0, 1, 3]])
    values = torch.tensor([1, 2, 4, -1, 3, 2, 5], dtype=torch.float32)
    spshape = (3, 4)
    x = torch.tensor([[[-1, -1, -1, -1, -1],
                       [6, 9, 1, 2, 7],
                       [2, 2, 2, 2, 1],
                       [1, 8, 2, 2, 5]],
                      [[1, 1, 1, 1, 1],
                       [-6, -9, -1, -2, -7],
                       [-2, -2, -2, -2, -1],
                       [-1, -8, -2, -2, -5]]], dtype=torch.float32)

    # Expected output
    expected = torch.tensor([[[7, 35, 11, 11, 21],
                              [-2, -2, -2, -2, -1],
                              [14, 55, 9, 11, 36]],
                             [[-7, -35, -11, -11, -21],
                              [2, 2, 2, 2, 1],
                              [-14, -55, -9, -11, -36]]], dtype=torch.float32)

    # Perform the test
    output = spmm_coo(indices, values, spshape, x)

    # Check if the output is as expected
    assert torch.allclose(output, expected), f"Expected {expected} but got {output}"


def test_batched_spmm_coo_2d_batched_vector():
    # Define test inputs
    indices = torch.tensor([[0, 0, 0, 1, 2, 2, 2],
                            [0, 2, 3, 2, 0, 1, 3]])
    values = torch.tensor([[1, 2, 4, -1, 3, 2, 5],
                           [2, 4, 8, -2, 6, 4, 10]], dtype=torch.float32)
    spshape = (3, 4)
    x = torch.tensor([[[-1, -1, -1, -1, -1],
                       [6, 9, 1, 2, 7],
                       [2, 2, 2, 2, 1],
                       [1, 8, 2, 2, 5]],
                      [[1, 1, 1, 1, 1],
                       [-6, -9, -1, -2, -7],
                       [-2, -2, -2, -2, -1],
                       [-1, -8, -2, -2, -5]]], dtype=torch.float32)

    # Expected output
    expected = torch.tensor([[[7, 35, 11, 11, 21],
                              [-2, -2, -2, -2, -1],
                              [14, 55, 9, 11, 36]],
                             [[-14, -70, -22, -22, -42],
                              [4, 4, 4, 4, 2],
                              [-28, -110, -18, -22, -72]]], dtype=torch.float32)

    # Perform the test
    output = spmm_coo(indices, values, spshape, x)

    # Check if the output is as expected
    assert torch.allclose(output, expected), f"Expected {expected} but got {output}"


def test_spmm_coo_invalid_shape():
    # Define test inputs with an invalid shape for x
    indices = torch.tensor([[0, 1], [1, 2], [0, 0]])
    values = torch.tensor([1, 1])
    spshape = (3, 3, 2)  # Incorrect shape: 3D but should be 2D in sparse
    x = torch.tensor([1, 2, 3])

    # Expect a ValueError to be raised
    with pytest.raises(ValueError):
        spmm_coo(indices, values, spshape, x)


def test_spmm_coo_invalid_spshape():
    # Define test inputs with an invalid shape for spshape
    indices = torch.tensor([[0, 1], [1, 2]])
    values = torch.tensor([1, 1])
    spshape = (3, 3)
    x = torch.tensor([1, 2])  # Incorrect shape: can not multiply

    # Expect a ValueError to be raised
    with pytest.raises(ValueError):
        spmm_coo(indices, values, spshape, x)
