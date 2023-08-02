from typing import (
    Optional, Tuple, Callable, Sequence, Union
)

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, device
from torch.nn import Module

from ..tools import mkfs, proj
from ..nntyping import VectorFunction


class TensorMapping(Module):
    """
    @brief A function whose input and output are tensors. The `forward` method\
           is not implemented, override it to build a function.
    """

    def get_device(self):
        """
        @brief Get the device of the first parameter in this module. Return `None`\
               if no parameters.
        """
        for param in self.parameters():
            return param.device

    ### features

    def mkfs(self, *input: Tensor, f_shape:Optional[Tuple[int, ...]]=None,
                   device: Optional[device]=None, **kwargs) -> Tensor:
        p = mkfs(*input, f_shape=f_shape, device=device)
        return self.__call__(p, **kwargs)

    __call__: Callable[..., Tensor]

    def fixed(self, idx: Sequence[int], value: Sequence[float]):
        """
        @brief Return a module wrapped from this, to make some input features fixed.\
               See `fealpy.pinn.modules.Fixed`.

        @param idx: Sequence[int]. The indices of features to be fixed.
        @param value: Sequence[int]. Values of data in fixed features.
        """
        assert len(idx) == len(value)
        return Fixed(self, idx, value)

    def extracted(self, *idx: int):
        """
        @brief Return a module wrapped from this. The output features of the wrapped module are extracted\
               from the original one. See `fealpy.pinn.modules.Extracted`.

        @param *idx: int. Indices of features to extract.
        """
        return Extracted(self, idx)

    ### numpy & mesh

    def from_numpy(self, ps: NDArray, device=None) -> Tensor:
        """
        @brief Accept numpy array as input, and return in Tensor type.

        @param ps: NDArray.
        @param device: torch.device | None. Specify the device when making tensor\
               from numpy array. Use the deivce of parameters in the module if\
               `None`. If no parameters in the module, use cpu by default.

        @return: Tensor.

        @note: This is a method with coordtype 'cartesian'.
        """
        pt = torch.from_numpy(ps)
        if device is None:
            device = self.get_device()
        return self.forward(pt.to(device=device))

    from_numpy.__dict__['coordtype'] = 'cartesian'

    def from_cell_bc(self, bc: NDArray, mesh) -> Tensor:
        """
        @brief From bc in mesh cells to outputs of the solution.

        @param bc: NDArray containing bc points. It may has a shape (m, TD+1) where m is the number\
                   of bc points and TD is the topology dimension of the mesh.

        @return: Tensor with shape (b, c, ...). Outputs in every bc points and every cells.\
                 In the shape (b, c, ...), 'b' represents bc points, 'c' represents cells, and '...'\
                 is the shape of the function output.
        """
        points = mesh.cell_bc_to_point(bc)
        return self.from_numpy(points)

    def get_cell_bc_func(self, mesh):
        """
        @brief Generate a barycentric function for the module, defined in the\
               given mesh cells.

        @return: A function with coordtype `barycentric`.
        """
        def func(bc: NDArray) -> NDArray:
            points = mesh.cell_bc_to_point(bc)
            return self.from_numpy(points).detach().numpy()
        func.__dict__['coordtype'] = 'barycentric'
        return func

    def estimate_error(self, other: VectorFunction, mesh=None, power: int=2, q: int=3,
                       split: bool=False, coordtype: str='b', squeeze: bool=False):
        """
        @brief Calculate error between the solution and `other` in finite element space `space`. Use this when all\
               parameters are in CPU memory.

        @param other: VectorFunction. The function(target) to be compared with.
        @param mesh: MeshLike, optional. A mesh in which the error is estimated. If `other` is a function in finite\
                     element space, use mesh of the space instead and this parameter will be ignored.
        @param power: int. Defaults to 2, which means to measure L-2 error by default.
        @param q: int. The index of quadratures.
        @param split: bool. Split error values from each cell if `True`, and the shape of return will be (NC, ...)\
                      where 'NC' refers to number of cells, and '...' is the shape of function output. If `False`,\
                      the output has the same shape to the funtion output. Defaults to `False`.
        @param coordtype: `'barycentric'`(`'b'`) or `'cartesian'`(`'c'`). Defaults to `'b'`. This parameter will be\
                          ignored if `other` has attribute `coordtype`.
        @param squeeze: bool. Defaults to `False`. Squeeze the function output before calculation.\
                        This is sometimes useful when estimating error for an 1-output network.

        @return: error.
        """
        from ...functionspace.Function import Function
        if isinstance(other, Function):
            mesh = other.space.mesh

        if mesh is None:
            raise ValueError("Param 'mesh' is required if the target is not a function in finite element space.")

        o_coordtype = getattr(other, 'coordtype', None)
        if o_coordtype is not None:
            coordtype = o_coordtype

        qf = mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        cellmeasure = mesh.entity_measure('cell')

        if coordtype in {'cartesian', 'c'}:

            ps = mesh.cell_bc_to_point(bcs)
            val = self.from_numpy(ps).detach().numpy()
            if squeeze:
                val = val.squeeze(-1)
            diff = np.abs(val - other(ps))**power

        elif coordtype in {'barycentric', 'b'}:

            val = self.from_cell_bc(bcs, mesh).detach().numpy()
            if squeeze:
                val = val.squeeze(-1)
            diff = np.abs(val - other(bcs))**power

        else:
            raise ValueError(f"Invalid coordtype '{coordtype}'.")

        e = np.einsum('q, qc..., c -> c...', ws, diff, cellmeasure)

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
        mesh_pt = [torch.from_numpy(x) for x in flat_mesh]
        pt_u: torch.Tensor = self.forward(torch.cat(mesh_pt, dim=1))
        u_plot: NDArray = pt_u.cpu().detach().numpy()
        assert u_plot.ndim == 2
        nf = u_plot.shape[-1]
        if nf <= 1:
            return u_plot.reshape(mesh[0].shape), mesh
        else:
            return [sub_u.reshape(mesh[0].shape) for sub_u in np.split(u_plot, nf, axis=-1)], mesh

    def add_surface(self, axes):
        pass


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


class Fixed(Solution):
    def __init__(self, net: Module,
                 idx: Sequence[int],
                 values: Sequence[float]
        ) -> None:
        """
        @brief Fix some input features of `net`, as a wrapped module.

        @param net: The original module.
        @param idx: Indices of features to be fixed.
        @param values: Values of fixed features.
        """
        super().__init__(net)
        self._fixed_idx = torch.tensor(idx, dtype=torch.long)
        self._fixed_value = torch.tensor(values, dtype=torch.float32).unsqueeze(0)

    def forward(self, p: Tensor):
        total_feature = p.shape[-1] + len(self._fixed_idx)
        size = p.shape[:-1] + (total_feature, )
        fixed_p = torch.zeros(size, dtype=torch.float, device=p.device)
        fixed_p[..., self._fixed_idx] = self._fixed_value

        feature_mask = torch.ones((total_feature, ), dtype=torch.bool)
        feature_mask[self._fixed_idx] = False
        fixed_p[..., feature_mask] = p

        return self.net.forward(fixed_p)


class Extracted(Solution):
    def __init__(self, net: Module,
                 idx: Sequence[int]
        ) -> None:
        """
        @brief Extract some output features of `net`, as a wrapped module.

        @param net: The original module.
        @param idx: Indices of output features to extract.
        """
        super().__init__(net)
        self._extracted_idx = torch.tensor(idx, dtype=torch.long)

    def forward(self, p: Tensor):
        return self.net.forward(p)[..., self._extracted_idx]


class Projected(Solution):
    def __init__(self, net: Module,
                 comps: Sequence[Union[None, Tensor, float]]) -> None:
        """
        @brief Project the input features of `net` into a sub space, as a wrapped module.\
               See `fealpy.pinn.tools.proj`.

        @param net: The original module.
        @param comps: Components in projected features.
        """
        super().__init__(net)
        self._comps = comps

    def forward(self, p: Tensor):
        inputs = proj(p, self._comps)
        return self.net.forward(inputs)
