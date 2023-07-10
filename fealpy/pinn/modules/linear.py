
from typing import Sequence
import math

import torch
from torch import Tensor
from torch.nn import Module, init
from torch.nn.parameter import Parameter


class StackStd(Module):
    """
    Standardize the inputs with sequence of centers and radius.\
    Stack the result in dim-1.

    @note:
    If shape of center and radius is (M, D) and shape of input is (N, D),
    then the shape of output is (N, M, D); where N is samples, M is number of
    centers to standardize and D is features.
    """
    def __init__(self, centers: Tensor, radius: float):
        super().__init__()
        self.centers = centers
        self.radius = radius

    def forward(self, p: Tensor):
        return (p[:, None, :] - self.centers[None, ...]) / self.radius


class MultiLinear(Module):
    def __init__(self, in_features: int, out_features: int, parallel: Sequence[int],
                 bias: bool=True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.ni = in_features
        self.no = out_features
        self.p = tuple(parallel)
        self.weight = Parameter(torch.empty(self.p + (self.ni, self.no), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(self.p + (self.no, ), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        fan_in = self.weight.shape[-2]
        gain = init.calculate_gain('leaky_relu', math.sqrt(5))
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor):
        ret = torch.einsum('...io, n...i -> n...o', self.weight, x)
        return self.bias[None, ...] + ret
