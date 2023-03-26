from typing import (
    Optional, Tuple, Callable, Sequence
)

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from torch.nn import Module
from torch.autograd import Variable

from .tools import mkfs
from .nntyping import VectorFunction, Operator, GeneralSampler


class TensorMapping(Module):
    """
    @brief A function whose input and output are tensors. The `forward` method\
           is not implemented, override it to build a function.
    """
    def _call_impl(self, *input: Tensor, f_shape:Optional[Tuple[int, ...]]=None,
                   **kwargs):
        p = mkfs(*input, f_shape=f_shape)
        return super()._call_impl(p, **kwargs)

    __call__: Callable[..., Tensor] = _call_impl

    def fixed(self, idx: Sequence[int], value: Sequence[float]):
        """
        @brief Return a module wrapped from this. The input of the wrapped module can provide\
               some features for the input of the original, and the rest features are fixed.

        @param idx: Sequence[int]. The indices of features to be fixed.
        @param value: Sequence[int]. Values of data in fixed features.
        """
        assert len(idx) == len(value)
        return _Fixed(self, idx, value)

    def extracted(self, *idx: int):
        """
        @brief Return a module wrapped from this. The output features of the wrapped module are extracted\
               from the original one.

        @param *idx: int. Indices of features to extract.
        """
        return _Extracted(self, idx)

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

    from_cell_bc.__dict__['coordtype'] = 'barycentric'

    def estimate_error(self, other: VectorFunction, space=None, split: bool=False,
                       coordtype: str='b', squeeze: bool=False):
        """
        @brief Calculate error between the solution and `other` in finite element space `space`.

        @param other: VectorFunction. The function to be compared with. If `other` is `Function`\
                      (functions in finite element spaces), there is no need to specify `space` and `fn_type`.
        @param space: FiniteElementSpace.
        @param split: bool. Split error values from each cell if `True`, and the shape of return will be (NC, ...)\
                      where 'NC' refers to number of cells, and '...' is the shape of function output. If `False`,\
                      the output has the same shape to the funtion output. Defaults to `False`.
        @param coordtype: `'barycentric'`(`'b'`) or `'cartesian'`(`'c'`). Defaults to `'b'`. This parameter will be\
                          ignored if `other` has attribute `coordtype`.
        @param squeeze: bool. Defaults to `False`. Squeeze the function output before calculation.\
                        This is sometimes useful when estimating error for an 1-output network.

        @return: error.
        """
        from ..functionspace.Function import Function
        if isinstance(other, Function):
            if space is not None:
                if other.space is not space:
                    print("Warning: Trying to calculate in another finite element space.")
            else:
                space = other.space

        o_coordtype = getattr(other, 'coordtype', None)
        if o_coordtype is not None:
            coordtype = o_coordtype

        if coordtype in {'cartesian', 'c'}:

            def f(ps):
                val = self(ps)
                if squeeze:
                    val = val.squeeze(-1)
                return (val - other(ps))**2
            f.__dict__['coordtype'] = 'cartesian'

        elif coordtype in {'barycentric', 'b'}:

            def f(bc):
                val = self.from_cell_bc(bc, space.mesh)
                if squeeze:
                    val = val.squeeze(-1)
                return (val - other(bc))**2
            f.__dict__['coordtype'] = 'barycentric'

        else:
            raise ValueError(f"Invalid coordtype '{coordtype}'.")

        e: np.ndarray = np.sqrt(space.integralalg.cell_integral(f))
        if split:
            return e
        return e.sum(axis=0)

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
                 idx: Sequence[int],
                 values: Sequence[float]
        ) -> None:
        super().__init__(net)
        self._fixed_idx = torch.tensor(idx, dtype=torch.long)
        self._fixed_value = torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    def forward(self, p: Tensor):
        total_feature = p.shape[-1] + len(self._fixed_idx)
        size = p.shape[:-1] + (total_feature, )
        fixed_p = torch.zeros(size, dtype=torch.float)
        fixed_p[..., self._fixed_idx] = self._fixed_value

        feature_mask = torch.ones((total_feature, ), dtype=torch.bool)
        feature_mask[self._fixed_idx] = False
        fixed_p[..., feature_mask] = p

        return self.net.forward(fixed_p)


class _Extracted(Solution):
    def __init__(self, net: Optional[Module],
                 idx: Sequence[int]
        ) -> None:
        super().__init__(net)
        self._extracted_idx = torch.tensor(idx, dtype=torch.long)

    def forward(self, p: Tensor):
        return self.net.forward(p)[..., self._extracted_idx]


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
        @param func: Operator.\
                     A function get x and u(x) as args. (e.g A pde or boundary condition.)
        @param target: Tensor or None.\
                       If `None`, the output will be compared with zero tensor.
        @param output_samples: bool. Defaults to `False`.\
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
