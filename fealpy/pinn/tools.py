from typing import Union, Optional, Tuple, Callable

import torch
from torch import Tensor, device


def mkfs(*inputs: Union[Tensor, float], f_shape: Optional[Tuple[int, ...]]=None,
         device: Optional[device]=None, requires_grad: bool=False) -> Tensor:
    """
    @brief Concatenate input tensors or floats into a single output tensor along the last dimension.

    @param *inputs: Union[Tensor, float]. Any number of tensors or floats to be concatenated\
                    into a single output tensor.
    @param f_shape: Tuple[int, ...], optional. If all the input(s) are float,\
                    each of them will be converted to a tensor with shape `f_shape`.\
                    If `f_shape` is not provided, the default shape is `(1,)`.
    @param device: device, optional. Specify the device when convert floats to tensors.
    @param requires_grad: bool, defaults to `False`.

    @return: The concatenated output tensor, with the size of the last dimension equal to the sum\
             of the sizes of the last dimensions of all input tensors or floats.

    @example:
    >>> import torch
    >>> a = torch.randn(2, 3)
    >>> b = 1.5
    >>> c = torch.tensor([[0.1], [0.2]])
    >>> mkfs(a, b, c)
    tensor([[ 1.4077, -0.6737,  0.7659,  1.5000,  0.1000],
            [ 2.4355, -1.8016, -0.5548,  1.5000,  0.2000]])
    """
    a = inputs[0]

    if len(inputs) == 1:
        if isinstance(a, Tensor):
            return a

        if f_shape is None:
            f_shape = (1, )
        return torch.tensor(float(a), device=device).expand(f_shape)

    b = inputs[1]

    if isinstance(a, Tensor):

        if not isinstance(b, Tensor):
            shape = tuple(a.shape[:-1]) + (1, )
            b = torch.tensor(b, device=device).expand(shape)

    else:

        if isinstance(b, Tensor):
            shape = tuple(b.shape[:-1]) + (1, )
            a = torch.tensor(a, device=device).expand(shape)

        else:
            if f_shape is None:
                f_shape = (1, )
            a = torch.tensor(float(a), device=device, requires_grad=requires_grad).expand(f_shape)
            b = torch.tensor(float(b), device=device, requires_grad=requires_grad).expand(f_shape)

    cated = torch.cat([a, b], dim=-1)

    if len(inputs) == 2:
        return cated
    return mkfs(cated, *inputs[2:], f_shape=f_shape)


def use_mkfs(func: Callable[..., Tensor]):
    """
    @brief Make a tensor function to receive multiple inputs.

    For example, if `f` is a function receiving a single tensor, like:
    ```
    def f(p: Tensor): ...
    ```
    where `p` is x-y position inputs with shape (m, 2). Use
    ```
    @use_mkfs
    def f(p: Tensor): ...
    ```
    and then `f` can be used like `f(x, y)`.
    """
    def wrapped(*input: Tensor, f_shape:Optional[Tuple[int, ...]]=None,
                device: Optional[device]=None, requires_grad: bool=False, **kwargs):
        p = mkfs(*input, f_shape=f_shape, device=device, requires_grad=requires_grad)
        return func(p, **kwargs)
    return wrapped
