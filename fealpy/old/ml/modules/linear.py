
from typing import Sequence, Union, Any

import torch
from torch import Tensor, float64, device
from torch.nn import Module, init, MSELoss
from torch.nn.parameter import Parameter
from torch.nn import functional as F


class Standardize(Module):
    """
    @brief: Standardize the inputs with sequence of centers and radius.\
    Stack the result with shape like (#Samples, #Centers, #Dims).
    """
    def __init__(self, centers: Tensor, radius: Union[Tensor, Any], device=None):
        """
        @param centers: Tensor with shape (M, GD).
        @param radius: Tensor with shape (M, GD) or (GD, ), or any other object\
               can be converted to tensors.

        @note:
        If shape of center and radius is (M, D) and shape of input is (N, D),
        then the shape of output is (N, M, D); where N is samples, M is number of
        centers to standardize and D is features.
        """
        super().__init__()
        self.centers = Parameter(centers.to(device=device), requires_grad=False)
        if not isinstance(radius, Tensor):
            rdata = torch.tensor(radius, dtype=centers.dtype, device=device)
        else:
            rdata = radius.to(device=device)
        self.radius = Parameter(rdata.expand(centers.shape).clone(), requires_grad=False)

    def forward(self, p: Tensor):
        return (p[:, None, :] - self.centers[None, :, :]) / self.radius[None, :, :]

    def inverse(self, p: Tensor):
        return p[:, None, :] * self.radius[None, :, :] + self.centers[None, :, :]

    def single(self, p: Tensor, ctr_idx: int):
        return (p - self.centers[None, ctr_idx, :]) / self.radius[None, ctr_idx, :]


class Distance(Module):
    """
    @brief Calculate the distances between inputs and source points.\
    Return with shape like (#Samples, #Sources).
    """
    def __init__(self, sources: Tensor, device=None) -> None:
        """
        @param sources: Tensor with shape (#Sources, #Dims).
        """
        super().__init__()
        self.sources = Parameter(sources.to(device=device), requires_grad=False)

    def forward(self, p: Tensor):
        return torch.norm(p[:, None, :] - self.sources[None, :, :], p=2,
                          dim=-1, keepdim=False)

    def gradient(self, p: Tensor):
        """
        @brief Return the gradient with shape (#Samples, #Sources, #Dims).
        """
        dis = self.forward(p)[:, :, None]
        return (p[:, None, :] - self.sources[None, :, :]) / dis


class MultiLinear(Module):
    def __init__(self, in_features: int, out_features: int, parallel: Sequence[int],
                 bias: bool=True, dtype=float64, device: device=None,
                 requires_grad=True) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.ni = in_features
        self.no = out_features
        self.p = tuple(parallel)
        self.weight = Parameter(
            torch.empty(self.p + (self.ni, self.no), **factory_kwargs),
            requires_grad=requires_grad
        )
        if bias:
            self.bias = Parameter(
                torch.empty(self.p + (self.no, ), **factory_kwargs),
                requires_grad=requires_grad
            )
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # fan_in = self.weight.shape[-2]
        # gain = init.calculate_gain('leaky_relu', math.sqrt(5))
        # std = gain / math.sqrt(fan_in)
        # bound = math.sqrt(3.0) * std
        bound = 1
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        if self.bias is not None:
            # bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor):
        ret = torch.einsum('...io, n...i -> n...o', self.weight, x)
        return self.bias[None, ...] + ret


class LeastSquare(Module):
    def __init__(self, ndof: int, gd: int=1, dtype=float64, device: device=None) -> None:
        """
        @brief Construct a least-square model.

        @param ndof: int. Number of variables.
        @param gd: int. Number of output features.
        """
        super().__init__()
        self.x_ = Parameter(torch.empty((ndof, gd), dtype=dtype, device=device))
        self.loss_fn = MSELoss(reduction='mean')
        self.reset_parameters()

    def reset_parameters(self):
        init.normal_(self.x_, 0.0, 0.1)

    def forward(self, A_: Tensor):
        # NOTE: A_ has shape (N, nf); x_ has shape (nf, gd)
        # The output has shape (N, gd)
        return F.linear(A_, self.x_.T).T

    def mse_loss(self, A_: Tensor, b_: Tensor):
        return self.loss_fn(self(A_), b_)
