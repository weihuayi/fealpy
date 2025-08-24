from typing import (
    Callable, Sequence, Union
)

from ...backend import backend_manager as bm
from ...typing import TensorLike
from numpy.typing import NDArray
import torch
from torch import Tensor, device, float64
from torch.nn import Module

from ..tools import proj
from fealpy.ml.nntyping import VectorFunction, TensorFunction


class TensorMapping(Module):
    """Base class for tensor-to-tensor mappings with extended functionality.
    
    Provides common operations for tensor mappings including device management,
    dimension operations, error estimation, and visualization tools. Designed as
    an abstract base class that should be subclassed with `forward` implemented.

    Attributes:
        _device: torch.device
            Stores the default device for tensor operations (optional)
    
    Methods:
        get_device(): Get the device of the first parameter or stored _device
        set_device(): Set the default device for the mapping
        last_dim(): Apply mapping while preserving all but last dimension
        diff(): Create difference mapping between two functions
        real(): Create mapping returning real part of complex output
        fixed(): Create mapping with fixed input features
        extracted(): Create mapping extracting specific output features
        from_numpy(): Convert numpy input to tensor and apply mapping
        from_cell_bc(): Evaluate at barycentric coordinates in mesh cells
        estimate_error(): Calculate error against reference function on mesh
        estimate_error_tensor(): Tensor-based error estimation
        estimate_error_func(): Functional error estimation
        meshgrid_mapping(): Evaluate mapping on meshgrid coordinates
        add_surface(): 3D surface plot of 2D input mapping
        add_pcolor(): 2D pseudocolor plot of 2D input mapping
    """

    def get_device(self) -> Union[device, None]:
        """Get the device of the first parameter in this module.
        
        Returns:
            torch.device or None: 
                Device of first parameter if exists, otherwise checks for _device attribute.
                Returns None if neither exists.
        """
        for param in self.parameters():
            return param.device

        return getattr(self, '_device', None)

    def set_device(self, device: device):
        """Set the default device for tensor operations.
        
        Parameters:
            device: torch.device
                Target device for tensor operations
                
        Note:
            Does not affect submodules' devices.
        """
        setattr(self, '_device', device)

    __call__: Callable[..., Tensor]

    ### dim operating

    def last_dim(self, p: Tensor):
        """Apply mapping while preserving all but last dimension.
        
        Parameters:
            p: Tensor
                Input tensor of shape (..., d) where d is input dimension
                
        Returns:
            Tensor: 
                Output tensor of shape (..., k) where k is output dimension,
                preserving all leading dimensions
        """
        origin_shape = p.shape[:-1]
        p = p.reshape(-1, p.shape[-1])
        val = self(p)
        return val.reshape(origin_shape + (val.shape[-1], ))

    ### module wrapping

    def diff(self, target: TensorFunction, solution2: TensorFunction=None):
        """Create a new mapping representing difference with target function.
        
        Parameters:
            target: TensorFunction
                Function to subtract from this mapping
                
        Returns:
            TensorMapping: 
                New mapping that computes self(p) - target(p)
        """
        return DiffSolution(self, target, fn3=solution2)

    def real(self, dtype):
        """Create mapping returning real part of complex output.
        
        Parameters:
            dtype: torch.dtype
                Data type for real conversion
                
        Returns:
            TensorMapping: 
                New mapping that computes real part of output
        """
        return RealSolution(self, dtype)

    def fixed(self, idx: Sequence[int], value: Sequence[float],
                 dtype=torch.float64):
        """Create mapping with fixed input features.
        
        Parameters:
            idx: Sequence[int]
                Indices of input features to fix
            value: Sequence[float]
                Values for fixed features
            dtype: torch.dtype, optional
                Data type for fixed values (default: float64)
                
        Returns:
            TensorMapping: 
                New mapping with specified input features fixed
        """
        assert len(idx) == len(value)
        return Fixed(self, idx, value, dtype=dtype)

    def extracted(self, *idx: int):
        """Create mapping extracting specific output features.
        
        Parameters:
            *idx: int
                Indices of output features to extract
                
        Returns:
            TensorMapping: 
                New mapping that outputs only specified features
        """
        return Extracted(self, idx)

    ### numpy & mesh

    def from_numpy(self, ps: NDArray, device=None, last_dim=False) -> Tensor:
        """Convert numpy array to tensor and apply mapping.
        
        Parameters:
            ps: NDArray
                Input numpy array
            device: torch.device, optional
                Target device (default: module's device)
            last_dim: bool, optional
                Whether to apply mapping on last dimension (default: False)
                
        Returns:
            Tensor: 
                Output tensor after applying mapping
                
        Note:
            This method has coordtype 'cartesian' attribute for compatibility.
        """
        pt = torch.from_numpy(ps)
        if device is None:
            device = self.get_device()
        if last_dim:
            return self.last_dim(pt.to(device=device))
        return self(pt.to(device=device))

    from_numpy.__dict__['coordtype'] = 'cartesian'

    def from_cell_bc(self, bc: NDArray, mesh, device=None) -> Tensor:
        """Evaluate mapping at barycentric coordinates in mesh cells.
        
        Parameters:
            bc: NDArray
                Barycentric coordinates with shape (m, TD+1)
            mesh: Mesh
                Computational mesh
            device: torch.device, optional
                Target device (default: module's device)
                
        Returns:
            Tensor: 
                Output values with shape (b, c, ...) where b is number of
                bc points and c is number of cells
        """
        points = mesh.cell_bc_to_point(bc)
        return self.from_numpy(points, device=device, last_dim=True)

    ### error

    def estimate_error(self, other: VectorFunction, mesh=None, power: int=2, q: int=3,
                       cell_type: bool=False, coordtype: str='b', squeeze: bool=False,
                       device=None, compare: str='real'):
        """Calculate error between this mapping and reference function.
        
        Parameters:
            other: VectorFunction
                Reference function to compare against
            mesh: Mesh, optional
                Mesh for error estimation (required if other not in FE space)
            power: int, optional
                L-error order (default: 2 for L2 error)
            q: int, optional
                Quadrature order (default: 3)
            cell_type: bool, optional
                Whether to return per-cell errors (default: False)
            coordtype: str, optional
                Coordinate type ('b' for barycentric, 'c' for cartesian) (default: 'b')
            squeeze: bool, optional
                Whether to squeeze output dimensions (default: False)
            device: torch.device, optional
                Computation device (default: module's device)
            compare: str
            'real' or 'imag', default: 'real'.

                
        Returns:
            Tensor: 
                Computed error measure
        """
        from fealpy.functionspace.function import Function

        if isinstance(other, Function):
            mesh = other.space.mesh

        if mesh is None:
            raise ValueError("Param 'mesh' is required if the target is not a function in finite element space.")
        assert compare in ('real', 'imag'), f"option is 'real' or 'imag', but got {compare}"

        o_coordtype = getattr(other, 'coordtype', None)
        if o_coordtype is not None:
            coordtype = o_coordtype

        qf = mesh.quadrature_formula(q, etype='cell')

        bcs, ws = qf.get_quadrature_points_and_weights()
        cellmeasure = mesh.entity_measure('cell')

        if coordtype in {'cartesian', 'c'}:

            ps = mesh.bc_to_point(bcs)
            val = self.from_numpy(ps.numpy(), device=device, last_dim=True).cpu()
    
            if squeeze:
                val = val.squeeze(-1)
            val_ture = other(ps)
            val = val.detach() if val.requires_grad else val

            # 统一形状
            ndim = len(val_ture.shape)
            if ndim == 2:  # 如果 val_ture 是 (N, M)
                val_ture = val_ture.unsqueeze(-1)  # -> (N, M, 1)
            elif ndim == 4:  # 如果 val_ture 是 (N, M, 1, 1)
                val_ture = val_ture.squeeze()  # -> (N, M, 1)
    
            # 检查最终形状是否匹配
            assert val.shape == val_ture.shape, f"Shape mismatch: val {val.shape}, val_ture {val_ture.shape}"
            if compare == "real":
                val = bm.real(val)
                val_ture = bm.real(val_ture)
            elif compare == 'imag':
                val = bm.imag(val)
                val_ture = bm.imag(val_ture)
            diff = bm.abs(val - val_ture)**power

        elif coordtype in {'barycentric', 'b'}:

            val = self.from_cell_bc(bcs, mesh, device=device).cpu().detach().numpy()
            if squeeze:
                val = val.squeeze(-1)
            val_ture = other(bcs)
            if compare == "real":
                val = bm.real(val)
                val_ture = bm.real(val_ture)
            elif compare == 'imag':
                val = bm.imag(val)
                val_ture = bm.imag(val_ture)
            diff = bm.abs(val - other(bcs))**power

        else:
            raise ValueError(f"Invalid coordtype '{coordtype}'.")

        e = bm.einsum('q, cq..., c -> c...', ws, diff, cellmeasure)
        if cell_type:
            return bm.pow(e, 1/power, out=e)
        
        return bm.pow(e.sum(axis=0), 1/power)

    def estimate_error_tensor(self, other: TensorFunction, mesh, *, power: int=2,
                              q: int=3, cell_type: bool=False, dtype=float64,solution2: TensorFunction= None):
        """Tensor-based error estimation between mappings.
        
        Parameters:
            other: TensorFunction
                Reference tensor function
            mesh: Mesh
                Computational mesh
            power: int, optional
                L-error order (default: 2)
            q: int, optional
                Quadrature order (default: 3)
            cell_type: bool, optional
                Whether to return per-cell errors (default: False)
            dtype: torch.dtype, optional
                Computation dtype (default: float64)
                
        Returns:
            Tensor: 
                Computed error measure
        """
        device = self.get_device()
        qf = mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        _cm = mesh.entity_measure('cell')

        if isinstance(_cm, float): # to tackle with the returns from cell_area of UniformMesh2d
            NC = mesh.number_of_cells()
            cellmeasure = torch.tensor(_cm, dtype=dtype, device=device).broadcast_to((NC, ))
        else:
            cellmeasure = _cm

        ps = mesh.bc_to_point(bcs)
        original_shape = ps.shape[:-1]
        ps = ps.reshape(-1, ps.shape[-1])

        if solution2 is not None:
            a = self(ps) + solution2(ps)
        else:
            a = self(ps)
        b = other(ps)
        if b.shape[-1] != 1:
            b = b[..., None]
        assert a.shape == b.shape, f"Shape mismatch: a {a.shape}, b {b.shape}"
        diff = torch.pow((a - b), power)
        diff = diff.reshape(original_shape + (diff.shape[-1], ))
        e = torch.einsum('q, cq..., c -> c...', ws, diff, cellmeasure)

        if cell_type:
            return torch.pow(e, 1/power, out=e)
        return torch.pow(e.sum(dim=0), 1/power)
    
    # def estimate_error_func(self, other: TensorFunction, mesh):
    #     """Functional error estimation via maximum difference at nodes.
        
    #     Parameters:
    #         other: TensorFunction
    #             Reference tensor function
    #         mesh: Mesh
    #             Computational mesh
                
    #     Returns:
    #         Tensor: 
    #             Maximum absolute difference at mesh nodes
    #     """
    #     device = self.get_device()
    #     node = mesh.entity('node')
    #     diff = torch.max(torch.abs(self(node) - other(node)))
    #     return diff

    ### plotting

    def meshgrid_mapping(self, *xi: Tensor, detach=True):
        """Evaluate mapping on meshgrid coordinates.
        
        Parameters:
            *xi: Tensor
                Coordinate vectors for meshgrid
            detach: bool, optional
                Whether to detach from computation graph (default: True)
                
        Returns:
            tuple: 
                (values, meshgrid) where values are either a single tensor or
                list of tensors for multi-output mappings
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
        """Add 3D surface plot of 2D input mapping.
        
        Parameters:
            axes: Axes3D
                Matplotlib 3D axes
            box: Sequence[float]
                Plotting domain [xmin, xmax, ymin, ymax]
            nums: Sequence[int]
                Grid resolution [nx, ny]
            dtype: torch.dtype, optional
                Computation dtype (default: float64)
            out_idx: Sequence[int], optional
                Output feature indices to plot (default: [0])
            edgecolor: str, optional
                Edge color (default: 'blue')
            linewidth: float, optional
                Edge linewidth (default: 0.0003)
            cmap: Colormap, optional
                Color mapping (default: RdYlBu_r)
            vmin: float, optional
                Color scale minimum
            vmax: float, optional
                Color scale maximum
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
        """Add 2D pseudocolor plot of 2D input mapping.
        
        Parameters:
            axes: Axes
                Matplotlib axes
            box: Sequence[float]
                Plotting domain [xmin, xmax, ymin, ymax]
            nums: Sequence[int]
                Grid resolution [nx, ny]
            dtype: torch.dtype, optional
                Computation dtype (default: float64)
            out_idx: int, optional
                Output feature index to plot (default: 0)
            vmin: float, optional
                Color scale minimum
            vmax: float, optional
                Color scale maximum
            cmap: Colormap, optional
                Color mapping (default: RdYlBu_r)
                
        Returns:
            QuadMesh: 
                Matplotlib collection object
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
        return torch.zeros_like(p[..., 0])


class Solution(TensorMapping):
    """
    A wrapper class that converts a tensor function into a TensorMapping object,
    providing a standardized interface for neural network solutions in PDE problems.

    Parameters:
        func: A callable TensorFunction - Must accept and return PyTorch tensors.
              Typically a torch.nn.Module instance or lambda function.

    Properties:
        net: Alias for the wrapped function (for backward compatibility)
        func: Access to the original function

    Methods:
        forward(p): Evaluates the wrapped function at input points p.

    Example:
        >>> net = nn.Sequential(nn.Linear(2, 64), nn.Tanh())
        >>> sol = Solution(net)  # Now compatible with fealpy's PDE solvers
        >>> points = torch.rand(100, 2)
        >>> outputs = sol(points)  # Forward pass through the network
    """
    def __init__(self, func: TensorFunction, complex: bool=False) -> None:
        super().__init__()
        self.__func = func
        self.complex = complex

    @property
    def net(self):
        """Getter for the wrapped network function"""
        return self.__func

    @property
    def func(self):
        """Getter for the original function"""
        return self.__func

    def forward(self, p: Tensor) -> Tensor:
        """
        Forward pass through the wrapped tensor function.

        Parameters:
            p: Input tensor of shape (..., d), where d is the input dimension.

        Returns:
            Tensor: Output tensor of shape (..., k), where k is the output dimension.
        """
        out = self.__func(p)
        if self.complex:
            if out.shape[-1] != 2:
                raise ValueError(f"Expected the network output shape to be (..., 2), but got {out.shape}")
            out = out[:, 0:1] + 1j * out[:, 1:2]

        return out


class DiffSolution(TensorMapping):
    """Mapping representing difference between two tensor functions.

    Parameters:
        fn1: TensorFunction
            First function to subtract from
        fn2: TensorFunction
            Second function to subtract
        fn3: TensorFunction, optional
            Third function to add (default: ZeroMapping)
    Return
        Tensor: Difference between fn1, fn2, and fn3 at input points p (fn1(p) - fn2(p) + fn3(p)).
    """
    def __init__(self, fn1: TensorFunction, fn2: TensorFunction, *, fn3:TensorFunction=None) -> None:
        super().__init__()
        self.__fn_1 = fn1
        self.__fn_3 = fn3 if fn3 is not None else ZeroMapping()
        self.__fn_2 = fn2

    def forward(self, p: Tensor):
        return self.__fn_1(p).flatten() + self.__fn_3(p).flatten() - self.__fn_2(p).flatten()
 

class RealSolution(TensorMapping):
    def __init__(self, fn: TensorFunction, dtype) -> None:
        super().__init__()
        self.__fn = fn
        self.dtype = dtype

    def forward(self, p: Tensor):
        return self.__fn(p.to(dtype=self.dtype)).real


class Fixed(Solution):
    """Mapping with fixed input features.
    
        Initialize fixed feature mapping.
        
        Parameters:
            func: TensorFunction
                Original mapping
            idx: Sequence[int]
                Indices of input features to fix
            values: Sequence[float]
                Values for fixed features
            dtype: torch.dtype, optional
                Data type for fixed values (default: float64)
    """
    def __init__(self, func: TensorFunction,
                 idx: Sequence[int],
                 values: Sequence[float],
                 dtype=torch.float64
        ) -> None:
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
    """Mapping extracting specific output features from another mapping.
        Initialize feature extraction mapping.
        
        Parameters:
            func: TensorFunction
                Original mapping
            idx: Sequence[int]
                Indices of output features to extract
    """
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
    """Mapping projecting input features into subspace before evaluation.
       Initialize projected input mapping.
        
        Parameters:
            func: TensorFunction
                Original mapping
            comps: Sequence[Union[None, Tensor, float]]
                Components defining projection subspace
    """
    def __init__(self, func: TensorFunction,
                 comps: Sequence[Union[None, Tensor, float]]) -> None:
        super().__init__(func)
        self._comps = comps

    def forward(self, p: Tensor):
        inputs = proj(p, self._comps)
        return self.func.forward(inputs)
