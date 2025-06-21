
from functools import reduce
from itertools import combinations_with_replacement
from typing import Sequence, Literal, Optional
import torch
from torch import Tensor, float64


def random_weights(m: int, n: int, dtype=float64, device=None) -> Tensor:
    """
    @brief Generate m random samples, where each sample has n features (n >= 2),\
    such that the sum of each feature is 1.0.

    @param m: The number of samples to generate.
    @param n: The number of features in each sample.

    @return: An ndarray with shape (m, n), where each row represents a random sample.

    @throws ValueError: If n < 2.
    """
    m, n = int(m), int(n)
    if n < 2:
        raise ValueError(f'Integer `n` should be larger than 1 but got {n}.')
    u = torch.zeros((m, n+1), dtype=dtype, device=device)
    u[:, n] = 1.0
    u[:, 1:n] = torch.sort(torch.rand(m, n-1, dtype=dtype), dim=1).values
    return u[:, 1:n+1] - u[:, 0:n]


def multi_index(p: int, n: int, device=None) -> Tensor:
    """Return multiple indices."""
    assert p >= 1, "`p` should be a positive integer."
    assert n >= 1, "`n` should be a positive integer."
    sep = torch.tensor(
        tuple(combinations_with_replacement(range(p), n-1)),
        dtype=torch.int
    )
    raw = torch.zeros((sep.shape[0], n+1), dtype=torch.int, device=device)
    raw[:, -1] = p - 1
    raw[:, 1:-1] = sep
    return raw[:, 1:] - raw[:, :-1]


def linspace_weights(p: int, n: int, dtype=float64, device=None) -> Tensor:
    """
    @brief Generate uniformly placed weights.
    """
    if p >= 2:
        return multi_index(p, n, device=device).to(dtype=dtype) / (p-1)
    elif p >= 1:
        ret = torch.zeros((1, n), dtype=dtype, device=device)
        ret[:, 0] = 1.0
        return ret
    else:
        raise ValueError("`p` should be a positive integer.")


def multiply(*bcs: Tensor, mode: Literal['dot', 'cross'],
             order: Optional[Sequence[int]]=None,
             dtype=float64, device=None) -> Tensor:
    """
    @brief Multiply bcs in different directions.

    @param *bcs: NDArray. Bcs operands for multiplying.
    @param mode: 'dot' or 'cross'.
    @param order: Sequence[int] | None.
    @param dtype: the default data type, for the output tensor([[1.]]) if no\
           `bcs` provided.
    @param device: the default device, for the output tensor([[1.]]) if no `bcs`\
           provided. `dtype` and `device` will be ignored if there are any `bcs`.

    @return: NDArray.
    """
    D = len(bcs)
    assert D <= 5
    if D == 0:
        return torch.ones((1, 1), dtype=dtype, device=device)
    NVC = reduce(lambda x,y: x*y, (bc.shape[-1] for bc in bcs), 1)
    desp1 = 'mnopq'
    desp2 = 'abcde'
    if mode == "dot":
        string = ", ".join(['m'+desp2[i] for i in range(D)])
        string += " -> m" + desp2[:D]
    elif mode == "cross":
        string = ", ".join([desp1[i]+desp2[i] for i in range(D)])
        string += " -> " + desp1[:D] + desp2[:D]
    bc = torch.einsum(string, *bcs).reshape(-1, NVC)
    if order is None:
        return bc
    else:
        return bc[:, order]
