from typing import (
    Optional, Callable, Sequence, Union
)

import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor, device, float64
from torch.nn import Module

from ..tools import proj
from ..nntyping import VectorFunction, TensorFunction


class TensorMapping(Module):
    """
    @brief A function whose input and output are tensors. The `forward` method\
           is not implemented, override it to build a function.
    """

    def get_device(self) -> Union[device, None]:
        """
        @brief Get the device of the first parameter in this module. Return `None`\
               if no parameters.

        If no submodule found, try to return `_device` attribute. This attribute\
        can be set by the `set_device` method.
        """
        for param in self.parameters():
            return param.device

        return getattr(self, '_device', None)

    def set_device(self, device: device):
        """
        @brief Set the default device. This can NOT set devices for submodules.
        """
        setattr(self, '_device', device)

    ### module

    __call__: Callable[..., Tensor]

    def diff(self, target: TensorFunction):
        """
        @brief Get a new module whose output is the difference.

        This is useful when drawing the difference.
        """
        return DiffSolution(self, target)

    def fixed(self, idx: Sequence[int], value: Sequence[float],
                 dtype=torch.float64):
        """
        @brief Return a module wrapped from this, to make some input features fixed.\
               See `fealpy.pinn.modules.Fixed`.

        @param idx: Sequence[int]. The indices of features to be fixed.
        @param value: Sequence[int]. Values of data in fixed features.
        @param dtype: dtype, optional.
        """
        assert len(idx) == len(value)
        return Fixed(self, idx, value, dtype=dtype)

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
                       cell_type: bool=False, coordtype: str='b', squeeze: bool=False):
        """
        @brief Calculate error between the solution and `other` in finite element space `space`.

        @param other: VectorFunction. The function(target) to be compared with.
        @param mesh: MeshLike, optional. A mesh in which the error is estimated. If `other` is a function in finite\
                     element space, use mesh of the space instead and this parameter will be ignored.
        @param power: int. Defaults to 2, which means to measure L-2 error by default.
        @param q: int. The index of quadratures.
        @param cell_type: bool. Split error values from each cell if `True`, and the shape of return will be (NC, ...)\
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
            val = self.from_numpy(ps).cpu().detach().numpy()
            if squeeze:
                val = val.squeeze(-1)
            diff = np.abs(val - other(ps))**power

        elif coordtype in {'barycentric', 'b'}:

            val = self.from_cell_bc(bcs, mesh).cpu().detach().numpy()
            if squeeze:
                val = val.squeeze(-1)
            diff = np.abs(val - other(bcs))**power

        else:
            raise ValueError(f"Invalid coordtype '{coordtype}'.")

        e = np.einsum('q, qc..., c -> c...', ws, diff, cellmeasure)

        if cell_type:
            return np.power(e, 1/power, out=e)
        return np.power(e.sum(axis=0), 1/power)

    def estimate_error_tensor(self, other: TensorFunction, mesh, *, power: int=2,
                              q: int=3, cell_type: bool=False, dtype=float64):
        """
        @brief Estimate error between the function and another tensor function.

        @param other: TensorFunction.
        @param mesh: Mesh. The mesh to estimate error.
        @param power: int, optional. The order of L-error, defaults to 2.
        @param q: int, optional. The index of quadratures, defualts to 3.
        @param cell_type: bool, optional.
        """
        device = self.get_device()
        qf = mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ws = torch.from_numpy(ws).to(dtype=dtype).to(device=device)
        _cm = mesh.entity_measure('cell')

        if isinstance(_cm, float): # to tackle with the returns from cell_area of UniformMesh2d
            NC = mesh.ds.number_of_cells()
            cellmeasure = torch.tensor(_cm, dtype=dtype, device=device).broadcast_to((NC, ))
        else:
            cellmeasure = torch.from_numpy(_cm).to(dtype=dtype).to(device=device)

        ps = torch.from_numpy(mesh.bc_to_point(bcs)).to(dtype=dtype).to(device=device)
        original_shape = ps.shape[:-1]
        ps = ps.reshape(-1, ps.shape[-1])
        diff = torch.pow(torch.abs(self(ps) - other(ps)), power)
        diff = diff.reshape(original_shape + (diff.shape[-1], ))
        e = torch.einsum('q, qc..., c -> c...', ws, diff, cellmeasure)

        if cell_type:
            return torch.pow(e, 1/power, out=e)
        return torch.pow(e.sum(dim=0), 1/power)

    ### plotting

    def meshgrid_mapping(self, *xi: NDArray):
        """
        @brief Calculate the function value in a meshgrid.

        @param *xi: ArrayLike. See `numpy.meshgrid`.

        @return: tensor, (X1, X2, ..., Xn). For output having more than one\
                 feature, return a list instead. Each element is a tensor,\
                 containing values of a feature in the meshgrid.
        """
        device_ = self.get_device()

        mesh = np.meshgrid(*xi)
        flat_mesh = [np.ravel(x).reshape(-1, 1) for x in mesh]

        mesh_pt = torch.cat([torch.from_numpy(x) for x in flat_mesh], dim=-1)
        pt_u: torch.Tensor = self.forward(mesh_pt.to(device=device_))
        u_plot: NDArray = pt_u.cpu().detach().numpy()
        assert u_plot.ndim == 2
        nf = u_plot.shape[-1]

        if nf <= 1:
            return u_plot.reshape(mesh[0].shape), mesh
        else:
            return [sub_u.reshape(mesh[0].shape) for sub_u in np.split(u_plot, nf, axis=-1)], mesh

    def add_surface(self, axes, box: Sequence[float], nums: Sequence[int],
                    out_idx: Sequence[int]=[0, ],
                    edgecolor='blue', linewidth=0.0003, cmap=None,
                    vmin=None, vmax=None):
        """
        @brief Draw a surface for modules having 2 input features.

        @param axes: Axes3D.
        @param box: Seuqence[float]. Box of the plotting area. Like `[0, 1, 0, 1]`.
        @param nums: Sequence[int]. Number of points in x and y direction.
        @param out_idx: Sequence[int]. Specify the output feature(s) to plot the\
               surface. Number of surfaces is equal to the number of output features.

        @returns: None.
        """
        from matplotlib import cm
        if cmap is None:
            cmap = cm.RdYlBu_r

        x = np.linspace(box[0], box[1], nums[0])
        y = np.linspace(box[2], box[3], nums[1])
        u, (X, Y) = self.meshgrid_mapping(x, y)
        if isinstance(u, list):
            for idx in out_idx:
                axes.plot_surface(X, Y, u[idx], cmap=cmap, edgecolor=edgecolor,
                                  linewidth=linewidth, antialiased=True,
                                  vmin=None, vmax=None)
        else:
            axes.plot_surface(X, Y, u, cmap=cmap, edgecolor=edgecolor,
                              linewidth=linewidth, antialiased=True,
                              vmin=None, vmax=None)

    def add_pcolor(self, axes, box: Sequence[float], nums: Sequence[int],
                    out_idx=0, vmin=None, vmax=None, cmap=None):
        """
        @brief Call pcolormesh for modules having 2 input features.

        @param axes: Axes.
        @param box: Sequence[float].
        @param nums: Sequence[int].
        @param out_index: int, optional. Specify the output feature to plot.

        @returns: matplotlib.collections.QuadMesh
        """
        from matplotlib import cm
        if cmap is None:
            cmap = cm.RdYlBu_r

        x = np.linspace(box[0], box[1], nums[0])
        y = np.linspace(box[2], box[3], nums[1])
        u, (X, Y) = self.meshgrid_mapping(x, y)
        if isinstance(u, list):
            return axes.pcolormesh(X, Y, u[out_idx], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            return axes.pcolormesh(X, Y, u, cmap=cmap, vmin=vmin, vmax=vmax)


class ZeroMapping(Module):
    def forward(self, p: Tensor):
        return torch.zeros_like(p)


class Solution(TensorMapping):
    """
    @brief Wrap a tensor function to be a TensorMapping object.
    """
    def __init__(self, net: Optional[TensorFunction]=None) -> None:
        super().__init__()
        if net:
            self.__net = net
        else:
            self.__net = ZeroMapping()

    @property
    def net(self):
        return self.__net

    def forward(self, p: Tensor) -> Tensor:
        return self.__net(p)


class DiffSolution(TensorMapping):
    def __init__(self, fn1: TensorFunction, fn2: TensorFunction) -> None:
        super().__init__()
        self.__fn_1 = fn1
        self.__fn_2 = fn2

    def forward(self, p: Tensor):
        return self.__fn_1(p) - self.__fn_2(p)


class Fixed(Solution):
    def __init__(self, net: TensorFunction,
                 idx: Sequence[int],
                 values: Sequence[float],
                 dtype=torch.float64
        ) -> None:
        """
        @brief Fix some input features of `net`, as a wrapped module.

        @param net: The original module.
        @param idx: Indices of features to be fixed.
        @param values: Values of fixed features.
        """
        super().__init__(net)
        self._fixed_idx = torch.tensor(idx, dtype=torch.long)
        self._fixed_value = torch.tensor(values, dtype=dtype).unsqueeze(0)

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
    def __init__(self, net: TensorFunction,
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
    def __init__(self, net: TensorFunction,
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
