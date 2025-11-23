from functools import reduce
from itertools import combinations_with_replacement
from typing import Sequence, Literal, Optional

from ...backend import backend_manager as bm
from ...typing import TensorLike


def random_weights(m: int, n: int, dtype=bm.float64, device=None) -> TensorLike:
    """Generate random samples with features summing to 1.0.
    
    This function creates m random samples, each with n features, where the sum
    of the features in each sample is normalized to 1.0. It's useful for generating
    probability distributions or weighted combinations.

    Parameters
        m : int
            The number of samples to generate.
        n : int
            The number of features in each sample. Must be >= 2.

    Returns
        samples : TensorLike
            A tensor of shape (m, n) containing the generated samples. Each row
            represents a sample where the sum of its elements is 1.0.

    Raises
        ValueError
            If `n` is less than 2.
    
    Notes
        The function uses a method involving sorting random numbers to ensure the
        sum constraint. It generates m vectors of n+1 elements, sets the first
        and last elements, sorts the middle elements, and then takes differences
        to ensure the sum is 1.0.

    Examples
        >>> samples = random_weights(5, 3)
        >>> samples.shape
        bm.Size([5, 3])
        >>> bm.allclose(samples.sum(dim=1), bm.ones(5))
        True
    """
    m, n = int(m), int(n)
    if n < 2:
        raise ValueError(f'Integer `n` should be larger than 1 but got {n}.')
    u = bm.zeros((m, n+1), dtype=dtype, device=device)
    u[:, n] = 1.0
    # u[:, 1:n] = bm.sort(bm.rand(m, n-1, dtype=dtype), dim=1).values
    u[:, 1:n] = bm.sort(bm.random.rand(m, n-1), axis=1)
    return u[:, 1:n+1] - u[:, 0:n]


def multi_index(p: int, n: int, device=None) -> TensorLike:
    """Return a tensor of multi-indices.

    This function generates a tensor containing all possible combinations with
    replacement of `n-1` indices chosen from the range [0, p-1]. The result is
    transformed into differences, effectively creating a set of indices useful
    for multi-dimensional interpolation or combinatorial problems.

    Parameters
        p : int
            The upper bound (exclusive) for the range of indices. Must be >= 1.
        n : int
            The number of elements in each combination. Must be >= 1.

    Returns
        indices : TensorLike
            A tensor of shape (num_combinations, n), where num_combinations is
            the number of combinations with replacement of (n-1) elements from p.
            Each row represents a multi-index.

    Raises
        AssertionError
            If `p` or `n` is less than 1.

    Notes
        The function uses `combinations_with_replacement` from itertools to
        generate the base combinations and then calculates the differences
        between consecutive elements, including an implicit 0 at the start and
        p-1 at the end for each combination.

    Examples
        >>> indices = multi_index(3, 2)
        >>> indices
        tensor([[0, 0],
                [0, 1],
                [1, 1]])
    """
    assert p >= 1, "`p` should be a positive integer."
    assert n >= 1, "`n` should be a positive integer."
    sep = bm.tensor(
        tuple(combinations_with_replacement(range(p), n-1)),
        dtype=bm.int64
    )
    raw = bm.zeros((sep.shape[0], n+1), dtype=bm.int64, device=device)
    raw[:, -1] = p - 1
    raw[:, 1:-1] = sep
    return raw[:, 1:] - raw[:, :-1]


def linspace_weights(p: int, n: int, rm_ends: bool=False, dtype=bm.float64, device=None) -> TensorLike:
    """Generate uniformly spaced weights.

    This function creates a tensor of weights that are uniformly distributed.
    If p >= 2, it uses the multi_index function to generate indices and scales
    them to create weights between 0 and 1. If p == 1, it returns a single
    weight vector with all zeros except the first element, which is 1.0.

    Parameters
        p(int): The number of points to divide the interval into. Must be >= 1.

        n(int): The number of features/weights in each sample. Must be >= 1.

        rm_ends(bool): If True, removes the first and last weight vectors from the result. Default is False.

        dtype(dtype): The desired data type of the output tensor. Default is bm.float64.

        device(device): The desired device of the output tensor (e.g., 'cpu' or 'cuda'). 
                       If None, uses the current default device. Default is None.

    Returns
        weights : TensorLike
            A tensor of shape (num_samples, n), where num_samples depends on p
            and n. If p >= 2, num_samples is the number of combinations with
            replacement of (n-1) elements from p. If p == 1, num_samples is 1.

    Raises
        ValueError
            If `p` is less than 1.

    Notes
        The function is useful for generating evenly spaced points or weights
        for interpolation or sampling tasks. When p=1, it acts as a special
        case returning a single, fully concentrated weight.

    Examples
        >>> weights = linspace_weights(3, 2)
        >>> weights
        tensor([[0.0000, 1.0000],
                [0.5000, 0.5000],
                [1.0000, 0.0000]])
        >>> weights = linspace_weights(1, 3)
        >>> weights
        tensor([[1., 0., 0.]])
    """
    if p >= 2:
        if rm_ends:
            weights = bm.astype(multi_index(p+2, n, device=device), dtype) / (p + 1)
            return weights[1:-1]
        return bm.astype(multi_index(p, n, device=device), dtype) / (p - 1)
    elif p >= 1:
        ret = bm.zeros((1, n), dtype=dtype, device=device)
        ret[:, 0] = 1.0
        return ret
    else:
        raise ValueError("`p` should be a positive integer.")


def multiply(*bcs: TensorLike, mode: Literal['dot', 'cross'],
             order: Optional[Sequence[int]]=None,
             dtype=bm.float64, device=None) -> TensorLike:
    """Multiply boundary conditions (bcs) in different directions.

    This function performs tensor multiplication (contraction) on a variable
    number of boundary condition tensors (bcs) using Einstein summation.
    It supports two modes: 'dot' (tensor dot product) and 'cross' (tensor
    product). The result can be reordered along the last dimension.

    Parameters
        *bcs : TensorLike
            Variable number of input tensors to be multiplied. Each tensor
            should have the same number of dimensions (rank).
        mode : Literal['dot', 'cross']
            The multiplication mode. 'dot' contracts adjacent dimensions,
            'cross' performs an outer product across dimensions.
        order : Optional[Sequence[int]], optional
            A sequence of integers specifying the desired order of the last
            dimension in the output tensor. If None, the default order is used.
            Default is None.
        dtype : dtype, optional
            The desired data type of the output tensor. Ignored if no bcs are
            provided. Default is bm.float64.
        device : device, optional
            The desired device of the output tensor. Ignored if no bcs are
            provided. Default is None.

    Returns
        result : TensorLike
            The result of the multiplication operation. If no bcs are provided,
            returns a tensor of ones with shape (1, 1). Otherwise, the shape
            depends on the input shapes and the mode.

    Raises
        AssertionError
            If the number of input tensors (D) exceeds 5.
        ValueError
            If `p` is less than 1.

    Notes
        The function uses `bm.einsum` for efficient tensor operations.
        The 'dot' mode is equivalent to sequentially applying `bm.tensordot`
        along adjacent dimensions, while 'cross' mode is similar to a generalized
        outer product. The `order` parameter allows reordering the resulting
        dimensions for specific applications.

    Examples
        >>> a = bm.rand(2, 3)
        >>> b = bm.rand(3, 4)
        >>> c = bm.rand(4, 5)
        >>> # Dot product mode
        >>> result_dot = multiply(a, b, c, mode='dot')
        >>> result_dot.shape
        bm.Size([2, 5])
        >>> # Cross product mode
        >>> result_cross = multiply(a, b, c, mode='cross')
        >>> result_cross.shape
        bm.Size([2, 3, 4, 5])
        >>> # With order
        >>> order = [2, 0, 1, 3] # Reorder dimensions of result_cross
        >>> result_reordered = multiply(a, b, c, mode='cross', order=order)
        >>> result_reordered.shape
        bm.Size([4, 2, 3, 5])
        >>> # No bcs
        >>> multiply()
        tensor([[1.]])
    """
    D = len(bcs)
    assert D <= 5
    if D == 0:
        return bm.ones((1, 1), dtype=dtype, device=device)
    NVC = reduce(lambda x,y: x*y, (bc.shape[-1] for bc in bcs), 1)
    desp1 = 'mnopq'
    desp2 = 'abcde'
    if mode == "dot":
        string = ", ".join(['m'+desp2[i] for i in range(D)])
        string += " -> m" + desp2[:D]
    elif mode == "cross":
        string = ", ".join([desp1[i]+desp2[i] for i in range(D)])
        string += " -> " + desp1[:D] + desp2[:D]
    bc = bm.einsum(string, *bcs).reshape(-1, NVC)
    if order is None:
        return bc
    else:
        return bc[:, order]
