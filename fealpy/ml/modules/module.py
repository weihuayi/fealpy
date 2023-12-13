from typing import (
    Callable, Sequence, Union
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

    __call__: Callable[..., Tensor]

    ### dim operating

    def last_dim(self, p: Tensor):
        """
        @brief Apply the model on the last dim/axis of the input `p`. Other dims\
               will be preserved.
        """
        origin_shape = p.shape[:-1]
        p = p.reshape(-1, p.shape[-1])
        val = self(p)
        return val.reshape(origin_shape + (val.shape[-1], ))

    ### module wrapping

    def diff(self, target: TensorFunction):
        """
        @brief Get a new module whose output is the difference.

        This is useful when drawing the difference.
        """
        return DiffSolution(self, target)

    def real(self, dtype):
        """
        @brief
        """
        return RealSolution(self, dtype)

    def fixed(self, idx: Sequence[int], value: Sequence[float],
                 dtype=torch.float64):
        """
        @brief Return a module wrapped from this, to make some input features fixed.\
               See `fealpy.ml.modules.Fixed`.

        @param idx: Sequence[int]. The indices of features to be fixed.
        @param value: Sequence[int]. Values of data in fixed features.
        @param dtype: dtype, optional.
        """
        assert len(idx) == len(value)
        return Fixed(self, idx, value, dtype=dtype)

    def extracted(self, *idx: int):
        """
        @brief Return a module wrapped from this. The output features of the wrapped module are extracted\
               from the original one. See `fealpy.ml.modules.Extracted`.

        @param *idx: int. Indices of features to extract.
        """
        return Extracted(self, idx)

    ### numpy & mesh

    def from_numpy(self, ps: NDArray, device=None, last_dim=False) -> Tensor:
        """
        @brief Accept numpy array as input, and return in Tensor type.

        @param ps: NDArray.
        @param device: torch.device | None. Specify the device when making tensor\
               from numpy array. Use the deivce of parameters in the module if\
               `None`. If no parameters in the module, use cpu by default.
        @param last_dim: bool. Apply the model on the last axis of the input if\
               `True`, and other axis will be preserved. Defaults to `False`.

        @return: Tensor.

        @note: This is a method with coordtype 'cartesian'.
        """
        pt = torch.from_numpy(ps)
        if device is None:
            device = self.get_device()
        if last_dim:
            return self.last_dim(pt.to(device=device))
        return self(pt.to(device=device))

    from_numpy.__dict__['coordtype'] = 'cartesian'

    def from_cell_bc(self, bc: NDArray, mesh, device=None) -> Tensor:
        """
        @brief From bc in mesh cells to outputs of the solution.

        @param bc: NDArray containing bc points. It may has a shape (m, TD+1) where m is the number\
                   of bc points and TD is the topology dimension of the mesh.

        @return: Tensor with shape (b, c, ...). Outputs in every bc points and every cells.\
                 In the shape (b, c, ...), 'b' represents bc points, 'c' represents cells, and '...'\
                 is the shape of the function output.
        """
        points = mesh.cell_bc_to_point(bc)
        return self.from_numpy(points, device=device, last_dim=True)

    ### error

    def estimate_error(self, other: VectorFunction, mesh=None, power: int=2, q: int=3,
                       cell_type: bool=False, coordtype: str='b', squeeze: bool=False,
                       device=None):
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
        @param coordtype: `'barycentric'`(`'b'`) or `'cartesian'`(`'c'`). The coordtype\
               of the target `other`. Defaults to `'b'`. This parameter will be\
               ignored if `other` has attribute `coordtype`.
        @param squeeze: bool. Defaults to `False`. Squeeze the function output before calculation.\
               This is sometimes useful when estimating error for an 1-output network.
        @param device: device | None. Use the device of the parameter in the model\
               if `None`.

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
            val = self.from_numpy(ps, device=device, last_dim=True).cpu().detach().numpy()
            if squeeze:
                val = val.squeeze(-1)
            diff = np.abs(val - other(ps))**power

        elif coordtype in {'barycentric', 'b'}:

            val = self.from_cell_bc(bcs, mesh, device=device).cpu().detach().numpy()
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

        @param other: TensorFunction. The target to compare with.
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

    def meshgrid_mapping(self, *xi: Tensor, detach=True):
        """
        @brief Calculate the function value in a meshgrid.

        @param *xi: Tensor. See `torch.meshgrid`.
        @param detach: bool, optional.

        @return: tensor, (X1, X2, ..., Xn). For output having more than one\
                 feature, return a list instead. Each element is a tensor,\
                 containing values of a feature in the meshgrid.
        """
        mesh = torch.meshgrid(*xi, indexing='ij')
        origin = mesh[0].shape
        flat_mesh = [torch.ravel(x).reshape(-1, 1) for x in mesh]
        mesh_pt = torch.cat(flat_mesh, dim=-1)
        mesh_pt = mesh_pt.to(device=self.get_device())

        val: Tensor = self(mesh_pt)

        if detach:
            val = val.cpu().detach()

        assert val.ndim in (1, 2)
        nf = val.shape[-1]

        if val.ndim == 1 or nf <= 1:
            return val.reshape(origin), mesh
        else:
            return [sub_u.reshape(origin) for sub_u in torch.split(val, 1, dim=-1)], mesh

    def add_surface(self, axes, box: Sequence[float], nums: Sequence[int],
                    dtype=float64,
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

        x = torch.linspace(box[0], box[1], nums[0], dtype=dtype)
        y = torch.linspace(box[2], box[3], nums[1], dtype=dtype)
        u, (X, Y) = self.meshgrid_mapping(x, y)
        if isinstance(u, list):
            for idx in out_idx:
                axes.plot_surface(X, Y, u[idx], cmap=cmap, edgecolor=edgecolor,
                                  linewidth=linewidth, antialiased=True,
                                  vmin=vmin, vmax=vmax)
        else:
            axes.plot_surface(X, Y, u, cmap=cmap, edgecolor=edgecolor,
                              linewidth=linewidth, antialiased=True,
                              vmin=vmin, vmax=vmax)

    def add_pcolor(self, axes, box: Sequence[float], nums: Sequence[int],
                   dtype=float64,
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

        x = torch.linspace(box[0], box[1], nums[0], dtype=dtype)
        y = torch.linspace(box[2], box[3], nums[1], dtype=dtype)
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
    def __init__(self, func: TensorFunction) -> None:
        super().__init__()
        self.__func = func

    @property
    def net(self):
        return self.__func

    @property
    def func(self):
        return self.__func

    def forward(self, p: Tensor) -> Tensor:
        return self.__func(p)


class DiffSolution(TensorMapping):
    def __init__(self, fn1: TensorFunction, fn2: TensorFunction) -> None:
        super().__init__()
        self.__fn_1 = fn1
        self.__fn_2 = fn2

    def forward(self, p: Tensor):
        return self.__fn_1(p) - self.__fn_2(p)


class RealSolution(TensorMapping):
    def __init__(self, fn: TensorFunction, dtype) -> None:
        super().__init__()
        self.__fn = fn
        self.dtype = dtype

    def forward(self, p: Tensor):
        return self.__fn(p.to(dtype=self.dtype)).real


class Fixed(Solution):
    def __init__(self, func: TensorFunction,
                 idx: Sequence[int],
                 values: Sequence[float],
                 dtype=torch.float64
        ) -> None:
        """
        @brief Fix some input features of `func`, as a wrapped module.

        @param func: The original module.
        @param idx: Indices of features to be fixed.
        @param values: Values of fixed features.
        """
        super().__init__(func)
        self._fixed_idx = torch.tensor(idx, dtype=torch.long)
        self._fixed_value = torch.tensor(values, dtype=dtype).unsqueeze(0)

    def forward(self, p: Tensor):
        total_feature = p.shape[-1] + len(self._fixed_idx)
        size = p.shape[:-1] + (total_feature, )
        fixed_p = torch.zeros(size, dtype=p.dtype, device=p.device)
        fixed_p[..., self._fixed_idx] = self._fixed_value

        feature_mask = torch.ones((total_feature, ), dtype=torch.bool)
        feature_mask[self._fixed_idx] = False
        fixed_p[..., feature_mask] = p

        return self.func.forward(fixed_p)


class Extracted(Solution):
    def __init__(self, func: TensorFunction,
                 idx: Sequence[int]
        ) -> None:
        """
        @brief Extract some output features of `func`, as a wrapped module.

        @param func: The original module.
        @param idx: Indices of output features to extract.
        """
        super().__init__(func)
        self._extracted_idx = torch.tensor(idx, dtype=torch.long)

    def forward(self, p: Tensor):
        return self.func.forward(p)[..., self._extracted_idx]


class Projected(Solution):
    def __init__(self, func: TensorFunction,
                 comps: Sequence[Union[None, Tensor, float]]) -> None:
        """
        @brief Project the input features of `func` into a sub space, as a wrapped module.\
               See `fealpy.pinn.tools.proj`.

        @param func: The original module.
        @param comps: Components in projected features.
        """
        super().__init__(func)
        self._comps = comps

    def forward(self, p: Tensor):
        inputs = proj(p, self._comps)
        return self.func.forward(inputs)
