import numpy as np
from scipy.special import factorial, comb
import string

def custom_next_permutation(arr, compare_function):
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
    arr : (NC, l, d) 
    alpha : (l, )
    """
    N = np.sum(alpha)
    s = string.ascii_lowercase[:N]
    ss = 'i'+',i'.join(s)
    s = ss+'->i'+s

    tup = (s, )
    for i in range(len(alpha)):
        for j in range(alpha[i]):
            tup = tup + (arr[:, i], )
    return np.einsum(*tup, optimize=False)

def symmetry_span_array(arr, alpha):
    M = span_array(arr, alpha)

    N = np.sum(alpha)
    idx = np.arange(N)
    idx1 = []
    for count, value in enumerate(alpha):
        idx1.extend([count] * value)
    ret = np.zeros_like(M)
    count = 0
    while True:
        ret += np.transpose(M, (0, ) + tuple([i+1 for i in idx])) 
        count += 1
        sss = custom_next_permutation(idx, lambda x, y : idx1[x]-idx1[y])
        if not sss:
            #ret *= np.prod(factorial(alpha))/factorial(np.sum(alpha))
            ret /= count
            break
    return ret

def symmetry_index(d, r):
    """
    @brief d 维 r 阶张量的对称部分，当张量拉长以后的索引
    """
    mapp = lambda x: np.array([int(ss) for ss in '0'*(r-len(np.base_repr(x,
        d)))+np.base_repr(x, d) ], dtype=np.int_)
    idx = np.array(list(map(mapp, np.arange(d**r))))
    flag = np.ones(len(idx), dtype=np.bool_)
    for i in range(r-1):
        flag = flag & (idx[:, i]<=idx[:, i+1])
    return np.where(flag)[0]



