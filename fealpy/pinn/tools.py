from typing import Union, Optional, Tuple, Callable

import torch
from torch import Tensor


def mkfs(*inputs: Union[Tensor, float], f_shape: Optional[Tuple[int, ...]]=None) -> Tensor:
    """
    Concatenate input tensors or floats into a single output tensor.

    Parameters:
    -----------
    *inputs: Union[Tensor, float]
        Any number of tensors or floats to be concatenated into a single output tensor.

    f_shape: Tuple[int, ...], optional (default=None)
        If all the input(s) are float, each of them will be converted to a tensor with shape `f_shape`.
        If `f_shape` is not provided, the default shape is `(1,)`.

    Returns:
    --------
    Tensor
        The concatenated output tensor, with the size of the last dimension equal to the sum
        of the sizes of the last dimensions of all input tensors or floats.

    Examples:
    ---------
    >>> import torch
    >>> a = torch.randn(2, 3)
    >>> b = 1.5
    >>> c = torch.tensor([[0.1], [0.2]])
    >>> result = mkfs(a, b, c)
    >>> result.shape
    (2, 5)
    """
    a = inputs[0]

    if len(inputs) == 1:
        if isinstance(a, Tensor):
            return a

        if f_shape is None:
            f_shape = (1, )
        return torch.tensor(float(a)).expand(f_shape)

    b = inputs[1]

    if isinstance(a, Tensor):

        if not isinstance(b, Tensor):
            shape = tuple(a.shape[:-1]) + (1, )
            b = torch.tensor(b).expand(shape)

    else:

        if isinstance(b, Tensor):
            shape = tuple(b.shape[:-1]) + (1, )
            a = torch.tensor(a).expand(shape)

        else:
            if f_shape is None:
                f_shape = (1, )
            a = torch.tensor(float(a)).expand(f_shape)
            b = torch.tensor(float(b)).expand(f_shape)

    cated = torch.cat([a, b], dim=-1)

    if len(inputs) == 2:
        return cated
    return mkfs(cated, *inputs[2:], f_shape=f_shape)


def use_mkfs(func: Callable[..., Tensor]):
    def wrapped(*input: Tensor, f_shape:Optional[Tuple[int, ...]]=None,
                **kwargs):
        p = mkfs(*input, f_shape=f_shape)
        return func(p, **kwargs)
    return wrapped
