from typing import (
    Optional, List, Literal, Union,
    Tuple, Callable
)

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import Module
from torch.autograd import Variable

from .tools import mkfs
from .nntyping import TensorFunction, VectorFunction, Operator, GeneralSampler


class TensorMapping(Module):
    """
    @brief A function whose input and output are tensors. The `forward` method
           is not implemented, override it to build a function.
    """
    def _call_impl(self, *input: Tensor, f_shape:Optional[Tuple[int, ...]]=None,
                   **kwargs):
        p = mkfs(*input, f_shape=f_shape)
        return super()._call_impl(p, **kwargs)

    __call__: Callable[..., Tensor] = _call_impl

    def fixed(self, feature_idx: List[int], value: List[float]):
        assert len(feature_idx) == len(value)
        return _Fixed(self, feature_idx, value)

    def from_cell_bc(self, bc: NDArray, mesh) -> NDArray:
        """
        @brief From bc in mesh cells to outputs of the solution.

        @param bc: NDArray containing bc points. It may has a shape (m, TD+1) where m is the number
                   of bc points and TD is the topology dimension of the mesh.

        @return: NDArray with shape (b, c, ...). Outputs in every bc points and every cells.
                 In the shape (b, c, ...), 'b' represents bc points, 'c' represents cells, and '...'
                 is the shape of the function value.
        """
        points = mesh.cell_bc_to_point(bc)
        points_tensor = torch.tensor(points, dtype=torch.float32)
        return self.forward(points_tensor).detach().numpy()

    def estimate_error(self, other: VectorFunction, space=None,
                       fn_type: Literal['bc', 'val']='bc', squeeze: bool=False):
        """
        @brief Calculate error between the solution and `other` in finite element space `space`.

        @param other: VectorFunction. The function to be compared with. If `other` is `Function`
                      (functions in finite element spaces), there is no need to specify `space` and `fn_type`.
        @param space: FiniteElementSpace.
        @param fn_type: 'bc' or 'val'. Defaults to 'bc'.
        @param squeeze: bool. Defaults to `False`. Squeeze the solution automatically if the last dim has shape (1).
                        This is sometimes useful when estimating error for an 1-output network.

        @return: error: float.
        """
        from ..functionspace.Function import Function
        if isinstance(other, Function):
            if space is not None:
                if other.space is not space:
                    print("Warning: Trying to calculate in another finite element space.")
            else:
                space = other.space

        if fn_type == 'val':

            def f(bc):
                val = self.from_cell_bc(bc, space.mesh)
                if squeeze:
                    val = val.squeeze(-1)
                return (val - other(space.mesh.cell_bc_to_point(bc)))**2

        elif fn_type == 'bc':

            def f(bc):
                val = self.from_cell_bc(bc, space.mesh)
                if squeeze:
                    val = val.squeeze(-1)
                return (val - other(bc))**2

        else:
            raise ValueError('Invalid function type.')

        return np.sqrt(space.integralalg.integral(f, False))

    def meshgrid_mapping(self, *xi: NDArray):
        """
        @brief Calculate the function value in a meshgrid.

        @param *xi: ArrayLike. See `numpy.meshgrid`.

        @return: outputs, (X1, X2, ..., Xn)
        """
        mesh = np.meshgrid(*xi)
        flat_mesh = [np.ravel(x).reshape(-1, 1) for x in mesh]
        mesh_pt = [Variable(torch.from_numpy(x).float(), requires_grad=True) for x in flat_mesh]
        pt_u: torch.Tensor = self.forward(torch.cat(mesh_pt, dim=1))
        u_plot: NDArray = pt_u.data.cpu().numpy()
        assert u_plot.ndim == 2
        nf = u_plot.shape[-1]
        if nf <= 1:
            return u_plot.reshape(mesh[0].shape), mesh
        else:
            return [sub_u.reshape(mesh[0].shape) for sub_u in np.split(u_plot, nf)], mesh


class ZeroMapping(Module):
    def forward(self, p: Tensor):
        return torch.zeros_like(p)


class Solution(TensorMapping):
    """
    @brief A function based on a submodule.
    """
    def __init__(self, net: Optional[Module]=None) -> None:
        """
        @brief Initialize a function based on a submodule.
               When `net` is a neural network module, this can be regarded as a sulotion of
               PDEs in PINN models.

        @param net: A Module in torch. Defaults to `None`.
        """
        super().__init__()
        if net:
            self.__net = net
        else:
            self.__net = ZeroMapping()

    @property
    def net(self):
        return self.__net

    def forward(self, p: Tensor):
        return self.__net(p)


class _Fixed(Solution):
    def __init__(self, net: Optional[Module],
                 idx: List[int],
                 values: List[float]
        ) -> None:
        super().__init__(net)
        self._fixed_idx = torch.tensor(idx, dtype=torch.long)
        self._fixed_value = torch.tensor(values, dtype=torch.float32).unsqueeze(-1)

    def forward(self, p: Tensor):
        total_feature = p.shape[-1] + len(self._fixed_idx)
        size = p.shape[:-1] + (total_feature, )
        fixed_p = torch.zeros(size, dtype=torch.float)
        fixed_p[..., self._fixed_idx] = self._fixed_value

        feature_mask = torch.ones((total_feature, ), dtype=torch.bool)
        feature_mask[self._fixed_idx] = False
        fixed_p[..., feature_mask] = p

        return self.net.forward(fixed_p)


class LearningMachine():
    """Neural network trainer."""
    def __init__(self, s: Solution, cost_function: Optional[Module]=None) -> None:
        self.__solution = s

        if cost_function:
            self.cost = cost_function
        else:
            self.cost = torch.nn.MSELoss(reduction='mean')

    @property
    def solution(self):
        return self.__solution


    def loss(self, sampler: GeneralSampler, func: Operator,
             target: Optional[Tensor]=None, output_samples: bool=False):
        """
        @brief Calculate loss value.

        @param sampler: Sampler.
        @param func: Operator.
                     A function get x and u(x) as args. (e.g A pde or boundary condition.)
        @param target: Tensor or None.
                       If `None`, the output will be compared with zero tensor.
        @param output_samples: bool. Defaults to `False`.
                               Return samples from the `sampler` if `True`.

        @note Arg `func` should be defined like:
        ```
            def equation(p: Tensor, u: TensorFunction) -> Tensor:
                ...
        ```
        Here `u` may be a function of `p`.
        """
        inputs = sampler.run()
        outputs = func(inputs, self.solution.forward)
        if not target:
            target = torch.zeros_like(outputs)
        ret: Tensor = self.cost(outputs, target)
        if output_samples:
            return ret, inputs
        return ret


def mkf(length: int, *inputs: Union[Tensor, float]):
    """Make features"""
    ret = torch.zeros((length, len(inputs)), dtype=torch.float32)
    for i, item in enumerate(inputs):
        if isinstance(item, Tensor):
            ret[:, i] = item[:, 0]
        else:
            ret[:, i] = item
    return ret


class _2dSpaceTime(Solution):
    _lef: Optional[TensorFunction] = None
    _le: float = 0
    _ref: Optional[TensorFunction] = None
    _re: float = 1
    _if: Optional[TensorFunction] = None

    def set_left_edge(self, x: float=0.0):
        self._le = float(x)
        return self._set_lef
    def _set_lef(self, bc: TensorFunction):
        self._lef = bc
        return bc
    def set_right_edge(self, x: float=1.0):
        self._re = float(x)
        return self._set_ref
    def _set_ref(self, bc: TensorFunction):
        self._ref = bc
        return bc
    def set_initial(self, ic: TensorFunction):
        self._if = ic
        return ic


class TFC2dSpaceTimeDirichletBC(_2dSpaceTime):
    """
    Solution on space(1d)-time(1d) domain with dirichlet boundary conditions,
    based on Theory of Functional Connections.
    """
    def forward(self, p: Tensor) -> Tensor:
        m = p.shape[0]
        l = self._re - self._le
        t = p[..., 0:1]
        x = p[..., 1:2]
        x_0 = torch.cat([torch.zeros_like(t), x], dim=1)
        t_0 = torch.cat([t, torch.ones_like(x)*self._le], dim=1)
        t_1 = torch.cat([t, torch.ones_like(x)*self._re], dim=1)
        return self._if(x)\
            + (self._re-x)/l * (self._lef(t)-self._if(mkf(m, self._le)))\
            + (x-self._le)/l * (self._ref(t)-self._if(mkf(m, self._re)))\
            + self.net(p) - self.net(x_0)\
            - (self._re-x)/l * (self.net(t_0)-self.net(mkf(m, 0, self._le)))\
            - (x-self._le)/l * (self.net(t_1)-self.net(mkf(m, 0, self._re)))
