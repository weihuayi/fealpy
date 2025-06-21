
import string
import numpy as np
from typing import Tuple

from itertools import combinations_with_replacement

from ..typing import TensorLike
from ..backend import backend_manager as bm
from .utils import tensor_basis


def generate_tensor_basis(basis: TensorLike, shape: Tuple[int, ...], dof_priority=True) -> TensorLike:
    """Generate tensor basis from scalar basis.

    Parameters:
        basis (Tensor): Basis of a scalar space, shaped (..., ldof).\n
        shape (Tuple[int, ...]): Shape of the dof.\n
        dof_priority (bool, optional): If True, the degrees of freedom are arranged\
        prior to their components. Defaults to True.

    Returns:
        Tensor: Basis of the tensor space, shaped (..., ldof*numel, *shape),\
        where numel is the number of elements in the shape.
    """
    kwargs = bm.context(basis)
    factor = tensor_basis(shape, **kwargs) # (numel, numel)
    # 计算张量积
    tb = bm.tensordot(basis, factor, axes=0) # (1, ldof ,ldof, numel, numel)
    ldof = basis.shape[-1]
    numel = factor.shape[0]

    if dof_priority:
        ndim = len(shape)
        # 如果 dof_priority 为 True，交换 ldof 和 numel 这两个维度的位置
        tb = bm.swapaxes(tb, -ndim-1, -ndim-2) # (1, ldof, numel, ldof, numel)

    tb = tb.reshape(basis.shape[:-1] + (numel*ldof,) + shape) # (1, ldof, ldof*numel, numel)

    return tb


def generate_tensor_grad_basis(grad_basis: TensorLike, shape: Tuple[int, ...], dof_priority=True) -> TensorLike:
    """Generate tensor grad basis from grad basis in scalar space.

    Parameters:
        grad_basis (Tensor): Gradient of basis of a scalar space, shaped (..., ldof, GD).\n
        shape (Tuple[int, ...]): Shape of the dof.\n
        dof_priority (bool, optional): If True, the degrees of freedom are arranged\
        prior to their components. Defaults to True.

    Returns:
        Tensor: Basis of the tensor space, shaped (..., ldof*numel, *shape, GD),\
        where numel is the number of elements in the shape.
    """
    factor = tensor_basis(shape, dtype=grad_basis.dtype)
    s0 = "abcde"[:len(shape)]
    tb = bm.einsum(f'...jz, n{s0} -> ...jn{s0}z', grad_basis, factor)
    ldof, GD = grad_basis.shape[-2:]
    numel = factor.shape[0]

    if dof_priority:
        ndim = len(shape)
        tb = bm.swapaxes(tb, -ndim-2, -ndim-3)

    return tb.reshape(grad_basis.shape[:-2] + (numel*ldof,) + shape + (GD,))

def custom_next_permutation(arr, compare_function):
    """
    @brief 生成下一个排列
    """
    n = len(arr)
    i = n - 2
    while i >= 0 and compare_function(arr[i], arr[i + 1]) >= 0:
        i -= 1
    if i == -1:
        # 如果没有找到降序的元素，说明当前排列已经是最大的排列
        return False
    # 从右向左查找第一个大于arr[i]的元素
    j = n - 1
    while compare_function(arr[j], arr[i]) <= 0:
        j -= 1
    # 交换arr[i]和arr[j]
    arr[i], arr[j] = arr[j], arr[i]
    # 反转arr[i+1:]，使其成为升序
    arr[i + 1:] = arr[i + 1:][::-1]
    return True

def span_array(arr, alpha):
    """
    @brief 计算 arr^alpha
    @param arr : (NC, l, d)
    alpha : (l, )
    """
    N = bm.sum(alpha)
    s = string.ascii_lowercase[:N]
    ss = 'i'+',i'.join(s)
    s = ss+'->i'+s

    tup = (s, )
    for i in range(len(alpha)):
        for j in range(alpha[i]):
            tup = tup + (arr[:, i], )
    return bm.einsum(*tup)

def symmetry_span_array(arr, alpha):
    """
    @brief 计算 arr^alpha 的对称部分
    @param arr : (NC, l, d)
    alpha : (l, )
    """
    M = span_array(arr, alpha)

    N = bm.sum(alpha)
    idx = [i for i in range(N)]
    idx1 = []
    for count, value in enumerate(alpha):
        idx1.extend([count] * value)
    ret = bm.zeros_like(M) # TODO 可以优化
    count = 0
    while True:
        ret += bm.transpose(M, (0, ) + tuple([i+1 for i in idx]))
        count += 1
        sss = custom_next_permutation(idx, lambda x, y : idx1[x]-idx1[y])
        if not sss:
            ret /= count
            break
    return ret

def symmetry_index(d, r, dtype=None, device=None):
    dtype = dtype if dtype is not None else bm.int32
    """
    @brief 将 d 维 r 阶张量拉长以后，其对称部分对应的索引和出现的次数
    """
    symidx0 = bm.tensor(list(combinations_with_replacement(range(d), r)),
                        dtype=dtype, device=device)
    coe = bm.flip(d**bm.arange(r, dtype=dtype, device=device))

    symidx = bm.einsum('ij,j->i', bm.astype(symidx0, bm.float64), bm.astype(coe, bm.float64))
    symidx = bm.astype(symidx, dtype)

    midx = bm.multi_index_matrix(r, d-1)
    #midx0 = bm.zeros_like(midx) 
    #for i in range(d):
    #    midx0[:, i] = bm.sum(symidx0 == i, axis=1) 
    #print(midx0-midx)
    #midx = midx0

    P = bm.concatenate([bm.tensor([1],device=device), bm.cumprod(bm.arange(r+1, device=device)[1:], axis=0)],
                       axis=0, dtype=dtype)
    num = P[r]/bm.prod(P[midx], axis=1, dtype=bm.float64)
    return symidx, num

def multi_index2d_to_index(midx):
    """
    @brief 计算二维多重指标的索引
    @param midx : (l, 3)
    @return idx : (l, )
    """
    idx = (a[..., 1]+a[..., 2])*(1+a[..., 1]+a[..., 2])//2 + a[..., 2]
    return idx

def multi_index3d_to_index(midx):
    """
    @brief 计算三维多重指标的索引
    @param midx : (l, 4)
    @return idx : (l, )
    """
    s1 = a[..., 1]+a[..., 2]+a[..., 3]
    s2 = a[..., 2]+a[..., 3]
    idx = s1*(1+s1)*(2+s1)//6 + s2*(s2+1)//2 + a[..., 3]
    return idx







