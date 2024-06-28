# test_coo_tensor.py
import torch
from torch.testing import assert_close
from fealpy.torch.sparse.coo_tensor import COOTensor


def test_dims_and_shape():
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 2, 0, 0, 2]])
    values = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.float64)
    sparse_shape = torch.tensor([3, 3])
    coo = COOTensor(indices, values, sparse_shape, is_coalesced=False)

    assert coo.ndim == 3
    assert coo.sparse_ndim == 2
    assert coo.dense_ndim == 1

    assert coo.shape == (2, 3, 3)
    assert coo.sparse_shape == (3, 3)
    assert coo.dense_shape == (2, )


def test_coalesce_with_values():
    # 创建一个未合并的COOTensor对象
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 2, 0, 0, 2]])
    values = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float64)
    sparse_shape = torch.tensor([3, 3])
    coo = COOTensor(indices, values, sparse_shape, is_coalesced=False)

    # 调用coalesce方法
    coalesced_coo = coo.coalesce()

    # 验证结果是否已合并
    assert coalesced_coo.is_coalesced

    # 验证值是否正确累积
    expected_indices = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]])
    expected_values = torch.tensor([1, 7, 3, 4], dtype=torch.float64)
    assert_close(coalesced_coo.indices, expected_indices)
    assert_close(coalesced_coo.values, expected_values)


def test_coalesce_without_values_accumulate():
    # 创建一个未合并的COOTensor对象，但没有值
    indices = torch.tensor([[0, 0, 1, 2, 0], [1, 2, 0, 0, 2]])
    values = None
    sparse_shape = torch.tensor([3, 3])
    coo = COOTensor(indices, values, sparse_shape, is_coalesced=False)

    # 调用coalesce方法，设置accumulate为True
    coalesced_coo = coo.coalesce(accumulate=True)

    # 验证结果是否已合并
    assert coalesced_coo.is_coalesced

    # 验证输出的值是否正确
    expected_indices = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]])
    expected_values = torch.tensor([1, 2, 1, 1])
    assert_close(coalesced_coo.indices, expected_indices)
    assert_close(coalesced_coo.values, expected_values)


def test_coalesce_without_values_not_accumulate():
    # 创建一个未合并的COOTensor对象，但没有值
    indices = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]])
    values = None
    sparse_shape = torch.tensor([3, 3])
    coo = COOTensor(indices, values, sparse_shape, is_coalesced=False)

    # 调用coalesce方法，设置accumulate为False
    coalesced_coo = coo.coalesce(accumulate=False)

    # 验证结果是否已合并
    assert coalesced_coo.is_coalesced

    # 验证输出的值是否为None
    assert coalesced_coo.values is None


def test_to_dense():
    indices = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 0]])
    values = torch.tensor([[1, 2, 3, 4], [6, 7, 8, 9]], dtype=torch.float64)
    sparse_shape = torch.tensor([3, 3])
    coo = COOTensor(indices, values, sparse_shape, is_coalesced=True)

    arr = coo.to_dense()
    assert arr.dtype == torch.float64
    assert_close(
        arr,
        torch.tensor([[[0, 1, 2],
                       [3, 0, 0],
                       [4, 0, 0]],
                      [[0, 6, 7],
                       [8, 0, 0],
                       [9, 0, 0]]], dtype=torch.float64)
    )
