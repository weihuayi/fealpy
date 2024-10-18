
from typing import Union, Dict, Optional, Callable, Sequence
from math import log

from numpy.typing import NDArray
import torch
from torch.nn import Parameter, Module, init
from torch import float64, device, relu


_dtype = torch.dtype
_device = torch.device
Tensor = torch.Tensor
Index = Union[int, Tensor, Sequence[int], slice]

_S = slice(None, None, None)

__all__ = [
    'Fractional',
    'FractionalWithHighcut',
    'MultiChannelFractional',
    'RegressiveFractional',
    'EigenNorm',
    'StackedFractional',
    'RegressionLoss'
]

class _EigenvalueBase(Module):
    n_dofs: int
    w: Tensor
    V: Tensor
    Vinv: Tensor

    def __init__(self, n_dofs: int, *, dtype: Optional[_dtype]=float64,
                 device: Union[_device, str, None]=None) -> None:
        super(_EigenvalueBase, self).__init__()
        kwargs = dict(dtype=dtype, device=device)
        self.n_dofs = n_dofs
        self.register_buffer('w', torch.empty((n_dofs, ), **kwargs))
        self.register_buffer('V', torch.empty((n_dofs, n_dofs), **kwargs))
        self.register_buffer('Vinv', torch.empty((n_dofs, n_dofs), **kwargs))

    def reset_operator(self):
        init.zeros_(self.w)
        init.orthogonal_(self.V)
        # NOTE: Data should be copied from V.T to Vinv. Otherwise, V will be
        # overriten by Vinv when loading the state dict.
        self.Vinv.copy_(self.V.T)

    def setup(self, w: Tensor, V: Tensor, Vinv: Optional[Tensor]=None):
        assert w.ndim == 1
        assert V.ndim == 2
        self.w.copy_(w)
        self.V.copy_(V)
        if Vinv is None:
            Vinv = self.V.T
        else:
            assert Vinv.ndim == 2
        self.Vinv.copy_(Vinv)

    def from_npz(self, filename: str, /):
        """Load a fractional operator from a .npz file.

        The file may contain the following keys:
        - 'w': A 1D tensor containing the eigen values.
        - 'v': A 2D tensor containing the eigen functions.
        - 'vinv': A 2D tensor containing the inverse of v, optional.
        - 'M': The 2D mass matrix, satisfying `vinv=v.T@M`, optional. Ignored if `vinv` is provided.
        """
        import numpy as np
        data: Dict[str, NDArray] = dict(np.load(filename))
        t_data = {k: torch.from_numpy(v) for k, v in data.items()}

        try:
            if 'vinv' in t_data:
                self.setup(t_data['w'], t_data['v'], t_data['vinv'])
            elif 'M' in t_data:
                Vinv = t_data['v'].T @ t_data['M']
                self.setup(t_data['w'], t_data['v'], Vinv)
            else:
                self.setup(t_data['w'], t_data['v'])
        except KeyError:
            raise KeyError(f"The file '{filename}' does not contain the required data.")

    def decompose(self, function: Tensor) -> Tensor:
        return torch.einsum('ik, ...k -> ...i', self.Vinv, function)

    def reconstruct(self, eigen_coef: Tensor) -> Tensor:
        return torch.einsum('ik, ...k -> ...i', self.V, eigen_coef)


class Fractional(_EigenvalueBase):
    def __init__(self, n_dofs: int, *, dtype: Optional[_dtype]=float64,
                 device: Union[_device, str, None]=None) -> None:
        super().__init__(n_dofs, dtype=dtype, device=device)
        kwargs = dict(dtype=dtype, device=device)
        self.gamma = Parameter(torch.zeros((), **kwargs))
        self.reset_operator()

    def initialize(self, gamma: float):
        """Initialize the order of the fractional operator."""
        with torch.no_grad():
            init.constant_(self.gamma, gamma)

    def matrix(self):
        V = self.V
        Vinv = self.Vinv
        L = torch.diag(torch.pow(self.w, self.gamma))
        return V @ L @ Vinv

    __call__: Callable[[Tensor], Tensor]

    def forward(self, gdvn: Tensor):
        return torch.einsum('ik, ...k -> ...i', self.matrix(), gdvn)


class FractionalWithHighcut(Fractional):
    def __init__(self, n_dofs: int, hc_slope=2., *, dtype=float64, device: device=None) -> None:
        super().__init__(n_dofs, dtype=dtype, device=device)
        kwargs = dict(dtype=dtype, device=device)
        self.hc = Parameter(torch.empty((), **kwargs), requires_grad=False)
        self.hc_slope = Parameter(torch.tensor(hc_slope, **kwargs), requires_grad=False)

    def initialize(self, s: float, hc: float):
        """Initialize the fractional operator order and the eigen value highcut."""
        super().initialize(s)
        with torch.no_grad():
            init.constant_(self.hc, hc)

    def matrix(self):
        V = self.V
        Vinv = self.Vinv
        hc = self.hc
        lam = self.w
        L = torch.diag(torch.pow(lam, self.gamma) * torch.pow(relu(lam/hc - 1) + 1, -self.hc_slope))
        return V @ L @ Vinv


class MultiChannelFractional(_EigenvalueBase):
    def __init__(self, n_dofs: int, n_channels: int, *, ch_index: Index=_S,
                 high_cut: bool=False, hc_slope=2.,
                 dtype=float64, device: device=None) -> None:
        super().__init__(n_dofs, dtype=dtype, device=device)
        assert n_channels > 0
        kwargs = dict(dtype=dtype, device=device)
        self.n_channels = n_channels
        self.ch_index = ch_index
        self.gamma = Parameter(torch.empty((n_channels, ), **kwargs))

        if high_cut:
            self.hc = Parameter(torch.empty((n_channels, ), **kwargs), requires_grad=False)
            self.hc_slope = Parameter(torch.tensor(hc_slope, dtype=dtype, device=device), requires_grad=False)
            self.matrix = self._transform_with_high_cut
        else:
            self.register_parameter('hc', None)
            self.register_parameter('hc_slope', None)
            self.matrix = self._transform
        self.reset_operator()
        self.reset_paramters()

    def reset_paramters(self):
        init.constant_(self.gamma, 0.0)
        if self.hc is not None:
            init.constant_(self.hc, self.w.max().item())

    def initialize(self, gamma: Sequence[float], hc: Optional[Sequence[float]]=None):
        """Initialize the fractional operator order and the eigen value highcut for each channel."""
        with torch.no_grad():
            self.gamma.copy_(torch.tensor(gamma, dtype=self.gamma.dtype, device=self.gamma.device))
            if hc is not None:
                if self.hc is None:
                    raise ValueError("The high cut has been disabled.")
                self.hc.copy_(torch.tensor(hc, dtype=self.hc.dtype, device=self.hc.device))

    def _transform(self):
        V = self.V
        Vinv = self.Vinv
        lam = self.w[None, :]
        switch = torch.zeros_like(self.gamma, requires_grad=False)
        switch[self.ch_index] = 1.0
        slope = (self.gamma * switch)[:, None]
        L = torch.pow(lam, slope)
        return torch.einsum('ij, cj, jk -> cik', V, L, Vinv)

    def _transform_with_high_cut(self):
        V = self.V
        Vinv = self.Vinv
        lam = self.w[None, :]
        hc = self.hc[self.ch_index, None]
        slope = self.gamma[self.ch_index, None]
        L = torch.pow(lam, slope) * torch.pow(relu(lam/hc - 1) + 1, -self.hc_slope)
        return torch.einsum('ij, cj, jk -> cik', V, L, Vinv)

    __call__: Callable[[Tensor], Tensor]

    def forward(self, data: Tensor) -> Tensor: # [n_channel, n_dof] -> [n_channel, n_dof]
        return torch.einsum('cik, ...ck -> ...ci', self.matrix(), data)


class RegressiveFractional(_EigenvalueBase):
    weights: Tensor
    gamma: Tensor

    def __init__(self, n_dofs: int, n_channels: int, *,
                 weight: Optional[bool]=True,
                 momentum: float=0.99,
                 eps: float=1e-6,
                 dtype: Optional[_dtype]=float64,
                 device: Union[_device, str, None]=None) -> None:
        super().__init__(n_dofs, dtype=dtype, device=device)
        kwargs = dict(dtype=dtype, device=device)
        self.n_channels = n_channels
        self.enable_weight = weight
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('weights', torch.empty((n_dofs, ), **kwargs))
        self.register_buffer('gamma', torch.empty((n_channels, ), **kwargs))
        self.reset_operator()
        self.reset_running_stats(weight)

    def setup(self, w: Tensor, V: Tensor, Vinv: Optional[Tensor] = None):
        super().setup(w, V, Vinv)
        self.reset_running_stats(self.enable_weight)

    def reset_running_stats(self, enable_weight: bool):
        self.gamma.zero_()
        log_eigen = torch.log10(self.w) # [n_dofs, ]
        log_eigen = log_eigen - log_eigen.mean() # scalar

        if enable_weight:
            weight = 1/(self.w*log(10))
            weight.sqrt_()
            W = torch.outer(weight, weight)

            torch.matmul(W, log_eigen, out=self.weights)
            self.weights.div_(
                log_eigen@W@log_eigen + self.eps
            )

        else:
            self.weights.copy_(log_eigen)
            self.weights.div_(
                torch.sum(log_eigen**2) + self.eps
            )

    def update(self, alpha: Tensor):
        b = self.momentum
        structure = alpha.shape[:-2]
        alpha_r = alpha.view(-1, self.n_channels, self.n_dofs).contiguous()
        # [N, n_channel, n_dof]
        log_alpha = alpha_r.abs_().log10_()
        if len(structure) != 0:
            log_alpha = log_alpha.mean(dim=0) # [n_channel, n_dof]
        mean_log_alpha = log_alpha.mean(dim=-1, keepdim=True) # [n_channel, 1]
        self.gamma.lerp_((mean_log_alpha - log_alpha)@self.weights, 1-b)

    def forward(self, data: Tensor):
        assert data.dim() >= 2
        alpha = self.decompose(data)

        if self.training:
            self.update(alpha)

        V = self.V
        lam = self.w[None, :]
        slope = self.gamma[:, None]
        L = torch.pow(lam, slope)
        return torch.einsum('...cj, ij, cj -> ...ci', alpha, V, L)


class EigenNorm(_EigenvalueBase):
    gain: Tensor

    def __init__(self, n_dofs: int, n_channels: int, *,
                 aggregate: str='max',
                 momentum: float=0.99,
                 eps: float=1e-6,
                 dtype: Optional[_dtype]=float64,
                 device: Union[_device, str, None]=None) -> None:
        super().__init__(n_dofs, dtype=dtype, device=device)
        kwargs = dict(dtype=dtype, device=device)
        self.n_channels = n_channels
        if aggregate not in ('max', 'mean'):
            raise ValueError(f"Unknown aggregate method: {aggregate}")
        self.aggregate = aggregate
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('gain', torch.empty((n_channels, n_dofs), **kwargs))
        self.reset_operator()
        self.reset_running_stats()

    def reset_running_stats(self) -> None:
        init.zeros_(self.gain)

    def update(self, alpha: Tensor) -> None:
        b = self.momentum
        alpha_r = alpha.view(-1, self.n_channels, self.n_dofs).contiguous()
        # [N, n_channel, n_dof]
        log_alpha = alpha_r.abs_().log10_()
        if self.aggregate == 'max':
            aggregated = log_alpha.max(dim=0, keepdim=False)[0]
        elif self.aggregate == 'mean':
            aggregated = log_alpha.mean(dim=0, keepdim=False) # [n_channel, n_dof]
        else:
            raise NotImplementedError(f"Unknown aggregate method: {self.aggregate}")
        self.gain.lerp_(-aggregated, 1-b)

    __call__: Callable[[Tensor], Tensor]

    def forward(self, data: Tensor) -> Tensor:
        assert data.dim() >= 2
        alpha = self.decompose(data)

        if self.training:
            self.update(alpha)

        V = self.V
        L = torch.pow(10., self.gain)
        return torch.einsum('...cj, ij, cj -> ...ci', alpha, V, L)


### Double fractional modules ###

class StackedFractional(_EigenvalueBase):
    def __init__(self, n_dofs: int, n_channels: int, *,
                 dtype: Optional[_dtype]=float64,
                 device: Union[_device, str, None]=None) -> None:
        super().__init__(n_dofs, dtype=dtype, device=device)
        kwargs = dict(dtype=dtype, device=device)
        self.s0 = Parameter(torch.empty((), **kwargs))
        self.s1 = Parameter(torch.empty((n_channels, ), **kwargs))
        self.reset_operator()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.s0, 0.)
        init.constant_(self.s1, 0.)

    @property
    def s(self) -> Tensor:
        return self.s0 + self.s1

    def update(self):
        with torch.no_grad():
            self.s0.add_(self.s1.mean())
            self.s1.sub_(self.s1.mean())

    def matrix(self):
        V = self.V
        Vinv = self.Vinv
        lam = self.w[None, :]
        slope = self.s[:, None]
        L = torch.pow(lam, slope)
        return torch.einsum('ij, cj, jk -> cik', V, L, Vinv)

    def forward(self, data: Tensor) -> Tensor: # [n_channel, n_dof] -> [n_channel, n_dof]
        if self.training:
            self.update()
        return torch.einsum('cik, ...ck -> ...ci', self.matrix(), data)


### Loss functions ###

class RegressionLoss(Module):
    x: Tensor
    weight: Tensor

    def __init__(self, n_dofs: int, n_channels: int, *,
                 weight=True,
                 dtype: Optional[_dtype]=float64,
                 device: Union[_device, str, None]=None) -> None:
        super().__init__()
        kwargs = dict(dtype=dtype, device=device)
        self.n_dofs = n_dofs
        self.n_channels = n_channels
        self.enable_weight = weight
        self.register_buffer('x', torch.empty((n_dofs, ), **kwargs))
        self.register_buffer('weight', torch.empty((n_dofs, ), **kwargs))

    def reset(self, w: Tensor) -> None:
        assert w.dim() == 1 and w.shape[0] == self.n_dofs
        self.x.copy_(torch.log10(w))

        if self.enable_weight:
            self.weight.copy_(1/(w*log(10)))
            self.weight.div_(self.weight.sum(dim=-1, keepdim=True))
        else:
            self.weight.fill_(1./self.n_dofs)

    def from_npz(self, filename: str):
        import numpy as np
        data: Dict[str, NDArray] = dict(np.load(filename))
        w = torch.from_numpy(data['w'])
        self.reset(w)
        return self

    # NOTE: shape of s: [channel, ] or []
    # shape of coef: [..., channel, dof]
    def forward(self, s: Tensor, coef: Tensor):
        log_coef = coef.detach().abs_().log10_() # [..., channel, dof]
        log_coef = log_coef.view(-1, self.n_channels, self.n_dofs) # [N, channel, dof]
        mean_x = self.x.mean()
        mean_y = log_coef.mean(dim=[0, 2]) # [channel, ]

        if s.ndim == 0:
            pred_mean_y = -s * (self.x - mean_x)[None, :] + mean_y[:, None] # [channel, dof]
        elif s.ndim == 1:
            assert s.shape[0] == self.n_channels
            pred_mean_y = -s[:, None] * (self.x - mean_x)[None, :] + mean_y[:, None] # [channel, dof]
        else:
            raise ValueError(f"Invalid shape of s: {s.shape}")

        loss = (log_coef - pred_mean_y[None, :, :]).square() # [N, channel, dof]
        loss = torch.einsum('nch, h -> nc', loss, self.weight)
        return loss.mean()
