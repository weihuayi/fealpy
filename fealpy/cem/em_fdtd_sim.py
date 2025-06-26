from matplotlib.projections import Axes3D
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
import numpy as np
from typing import Tuple, Optional, Callable, Any, Union, List
from math import pi
from ..backend import backend_manager as bm

import warnings

class EMFDTDSim:
    # electromagnetics FDTD method simulation
    def __init__(self, mesh, NT=200, R=None, permittivity=1, permeability=1,dt=None,device=None):

        self.mesh = mesh
        self.device = device if device is not None else mesh.device

        self.c = 299792458.0
        self.mu0: float = 4e-7 * pi
        self.epsilon0 = 8.854187817e-12

        self.h = mesh.h[0]
        self.R = R if R is not None else self.c * dt/self.h
        self.dt = self.h * self.R / self.c if dt is None else dt

        if self.R > 1 / (mesh.TD ** 0.5):
            warnings.warn("Time step is too large, which may lead to instability!", UserWarning)
        self.NT = NT
        self.dim = mesh.TD

        self._register_update_methods()
        self.permittivity = permittivity
        self.permeability = permeability
        self.init_perm_matrix()
        self.boundary_condition = 'PEC'

        self.eta0: float = self.mu0 * self.c
        # 存储源项信息
        self.sources = []
        # 存储物体信息
        self.objects = []
        self.E = {'x': None, 'y': None, 'z': None}
        self.H = {'x': None, 'y': None, 'z': None}


#################################### 网格 #############################################        
    def cell_location(self, p):
        """
        Determine the index of the cell containing each point in a uniform 2D or 3D mesh.

        Args:
            p (Tensor): Array of shape (..., TD), where TD is the mesh's topological dimension.
                        Each entry represents a point in physical coordinates.

        Returns:
            Tuple of integer arrays (n0, n1, ..., n{TD-1}), where each array contains
            the index of the containing cell along each axis.
        """
        mesh = self.mesh
        TD = mesh.TD  # Topological dimension (must be 2 or 3)
        if TD not in (2, 3):
            raise ValueError(f"Unsupported mesh dimension: TD = {TD}. Only 2D or 3D supported.")

        ftype = mesh.ftype  # Floating point precision 
        itype = mesh.itype  # Integer type for indices 
        device = self.device  # Compute device (e.g., 'cpu' or 'cuda')

        # Mesh spacing and origin as tensors
        h = bm.array(mesh.h, dtype=ftype, device=device)       # Grid spacing per axis
        origin = mesh.origin  # Grid origin

        # Convert input coordinates to proper type and shift to mesh frame
        p = bm.asarray(p, dtype=ftype, device=device)
        v = p - origin  # Shifted point coordinates

        # Compute cell indices with floor rounding to handle boundary points,
        # epsilon added to account for numerical noise near boundary
        epsilon = 1e-10
        floored = bm.floor(v / h + epsilon)
        indices = bm.astype(floored, itype, device=device)

        return tuple(indices[..., d] for d in range(TD))

    def node_location(self, p):
        """
        Locate the nearest grid nodes for a set of points in a uniform mesh.

        Args:
            p (Tensor): Array of shape (..., TD), where TD is the topological dimension.
                        Each entry represents a point in space.

        Returns:
            Tuple of integer arrays (n0, n1, ..., n{TD-1}), where each array contains
            the index of the nearest node along each axis.
        """
        mesh = self.mesh
        TD = mesh.TD  # Topological dimension (2 or 3)
        if TD not in (2, 3):
            raise ValueError(f"Only 2D or 3D meshes are supported. Got TD = {TD}")

        ftype = mesh.ftype  # Floating point type for computation
        itype = mesh.itype  # Integer type for index values
        device = mesh.device  # Compute device (e.g., 'cpu' or 'cuda')

        # Mesh spacing and origin in each dimension
        h = mesh.h
        origin = mesh.origin

        # Convert input points to appropriate dtype and device, then shift to mesh frame
        p = bm.asarray(p, dtype=ftype, device=device)
        v = p - origin  # Shifted coordinates relative to mesh origin

        # Traditional round-half-up rule:
        # Round positive values up at .5, and negative values down at -.5
        rounded = bm.where(v >= 0, bm.floor(v / h + 0.5), bm.ceil(v / h - 0.5))

        # Cast rounded indices to integer tensor and split by dimension
        indices = bm.astype(rounded, itype, device=device)

        return tuple(indices[..., d] for d in range(TD))


    def interpolation(self, f, intertype='node'):
        """
        Interpolate the function f at specified mesh locations.

        Args:
            f (Callable): Function to interpolate, accepts (..., TD) tensor input.
            intertype (str): Interpolation target, one of:
                            {'node', 'cell', 'facex', 'facey', 'facez', 'face',
                            'edgex', 'edgey', 'edgez', 'edge'}.
        Returns:
            Tensor or tuple of Tensors: Interpolated values at mesh locations.
        """
        mesh = self.mesh

        if intertype == 'cell':
            F = f(mesh.cell_barycenter())
        elif intertype == 'node':
            node = mesh.node.reshape(
            mesh.nx + 1, mesh.ny + 1,
            getattr(mesh, 'nz', 0) + 1, mesh.TD
        ).squeeze()
            F = f(node)
        elif intertype == 'facex':
            F = f(mesh.facex_barycenter())
        elif intertype == 'facey':
            F = f(mesh.facey_barycenter())
        elif intertype == 'facez':
            F = f(mesh.facez_barycenter())
        elif intertype == 'face':
            xbc, ybc, zbc = mesh.face_barycenter()
            F = f(xbc), f(ybc), f(zbc)
        elif intertype == 'edgex':
            F = f(mesh.edgex_barycenter())
        elif intertype == 'edgey':
            F = f(mesh.edgey_barycenter())
        elif intertype == 'edgez':
            F = f(mesh.edgez_barycenter())
        elif intertype == 'edge':
            bary = mesh.edge_barycenter()
            F = tuple(f(b) for b in bary)

        else:
            raise ValueError(f"Unknown interpolation type: {intertype}")

        return F

#################################### 初始电磁场 #############################################  


    def initialize_field(self, field_name: str, func, times=None):
        """
        Initialize a Yee-grid EM field component (vectorized, 2D/3D) and optionally record true solutions.

        Args:
            field_name (str): Component name, e.g. 'E_x', 'H_y', 'H_z'.
            func (callable): 2D → func(X, Y, t), 3D → func(X, Y, Z, t).
            times (int or iterable of float, optional):
                If int N: save solutions at t0 + k*dt for k=0…N.
                If iterable: use these explicit time points.

        Returns:
            Tensor: Field values at the base time (t = 0 for E, t = –dt/2 for H).
        """
        mesh = self.mesh
        TD = mesh.TD
        dt = self.dt
        device = self.device

        # prepare storage for true solutions
        self.true_solutions = getattr(self, 'true_solutions', {})

        # parse field_name
        kind, comp = field_name.split('_')
        if kind not in {'E', 'H'}:
            raise ValueError("field_name must start with 'E' or 'H'")
        field_dict = self.E if kind == 'E' else self.H

        # set base time: E at integer step, H at half-step
        base_t = 0.0 if kind == 'E' else -0.5 * dt

        # obtain nodal coordinates (..., TD)
        node = mesh.node.reshape(
            mesh.nx + 1, mesh.ny + 1,
            getattr(mesh, 'nz', 0) + 1, mesh.TD
        ).squeeze()

        # compute sampling coordinates for this component
        if TD == 2:
            if kind == 'E' and comp == 'z':
                coords = node
            elif kind == 'H' and comp == 'x':
                # average along x-edges
                coords = (node[:, :-1] + node[:, 1:]) * 0.5
            elif kind == 'H' and comp == 'y':
                # average along y-edges
                coords = (node[:-1, :] + node[1:, :]) * 0.5
            else:
                raise ValueError("2D supports only 'E_z', 'H_x', 'H_y'")
            X, Y = coords[..., 0], coords[..., 1]
            arr = func(X, Y, base_t)

        else:  # TD == 3
            def avg(a, axis):
                # center-average along given axis
                sl = [slice(None)] * a.ndim
                fwd = sl.copy(); fwd[axis] = slice(1, None)
                bwd = sl.copy(); bwd[axis] = slice(None, -1)
                return 0.5 * (a[tuple(fwd)] + a[tuple(bwd)])

            axis = 'xyz'.index(comp)
            if kind == 'E':
                coords = avg(node, axis)
            else:
                coords = node
                for ax in sorted({0,1,2} - {axis}, reverse=True):
                    coords = avg(coords, ax)
            X, Y, Z = coords[..., 0], coords[..., 1], coords[..., 2]
            arr = func(X, Y, Z, base_t)

        # record true-solution time series if requested
        if times is not None:
            if isinstance(times, int):
                t_pts = base_t + dt * bm.arange(times + 1, device=device)
            else:
                t_pts = bm.asarray(list(times), dtype=float, device=device)

            if TD == 2:
                data = bm.stack([func(X, Y, t) for t in t_pts], axis=0)
            else:
                data = bm.stack([func(X, Y, Z, t) for t in t_pts], axis=0)

            self.true_solutions[field_name] = data

        # store and return initial field
        field_dict[comp] = arr
        return arr


    def get_field_matrix(self, etype='node', axis=None):
        """
        Return zero-initialized field arrays on a Yee grid for nodes, cells, faces, or edges.

        Args:
            etype (str): 'node', 'cell', 'face', or 'edge'.
            axis (str or None): 'x', 'y', 'z' to select one direction; None for all.

        Returns:
            bm.ndarray or tuple: Array(s) of shape matching mesh layout for the requested type.
        """
        mesh = self.mesh
        TD = mesh.TD
        dtype = mesh.ftype
        nx, ny = mesh.nx, mesh.ny
        nz = getattr(mesh, 'nz', 0)

        # helper to allocate zeros on correct device
        def Z(shape):
            return bm.zeros(shape, dtype=dtype, device=self.device)

        if TD == 2:
            if etype == 'node':
                return Z((nx+1, ny+1))
            if etype == 'cell':
                return Z((nx, ny))
            if etype == 'edge':
                # both x- and y-edges
                if axis is None:
                    return Z((nx+1, ny)), Z((nx, ny+1))
                if axis == 'x':
                    return Z((nx+1, ny))
                if axis == 'y':
                    return Z((nx, ny+1))

        else:  # TD == 3
            if etype == 'node':
                return Z((nx+1, ny+1, nz+1))
            if etype == 'cell':
                return Z((nx, ny, nz))
            if etype == 'face':
                # face centers perpendicular to each axis
                if axis is None:
                    return Z((nx+1, ny, nz)), Z((nx, ny+1, nz)), Z((nx, ny, nz+1))
                if axis == 'x':
                    return Z((nx+1, ny, nz))
                if axis == 'y':
                    return Z((nx, ny+1, nz))
                if axis == 'z':
                    return Z((nx, ny, nz+1))
            if etype == 'edge':
                # edge centers along each axis
                if axis is None:
                    return Z((nx, ny+1, nz+1)), Z((nx+1, ny, nz+1)), Z((nx+1, ny+1, nz))
                if axis == 'x':
                    return Z((nx, ny+1, nz+1))
                if axis == 'y':
                    return Z((nx+1, ny, nz+1))
                if axis == 'z':
                    return Z((nx+1, ny+1, nz))

        raise ValueError(f"Invalid etype='{etype}' or axis='{axis}' for TD={TD}")


    def init_field_matrix(self, type = None ,axis = None):
        dim = self.mesh.TD

        if dim == 2:
            if type == 'E':
                return self.get_field_matrix(etype='node', axis=axis)
            elif type == 'H':
                return self.get_field_matrix(etype='edge', axis=axis)
        elif dim == 3:
            if type == 'E':
                return self.get_field_matrix(etype='edge', axis=axis)
            elif type == 'H':
                return self.get_field_matrix(etype='face', axis=axis)
            
    def _init_fields_dict(self, field, comps, num_frames, axis_type=True):
        """
        Initialize dictionaries for storing field components and their time frames.

        Args:
            field (str): 'E' or 'H' indicating electric or magnetic field.
            comps (iterable): Component keys, e.g. ['x', 'y', 'z'].
            num_frames (int): Number of time frames to allocate (0 means no history).
            axis_type (bool): If True, store (prev, curr) tuples; if False, store only curr.

        Returns:
            Tuple:
                fields (dict): {comp: tensor or (tensor, tensor)} initial field arrays.
                data (dict):   {comp: tensor[num_frames, ...]} time-series storage.
        """
        fields = {}
        data = {}

        # For each component, determine initial arrays f0 (prev) and f1 (curr)
        for comp in comps:
            # Existing field values take precedence
            if field == 'E' and self.E[comp] is not None:
                # Scale E by impedance for prev/curr
                f0 = self.E[comp] / self.eta0
                f1 = self.E[comp] / self.eta0
            elif field == 'H' and self.H[comp] is not None:
                f0 = self.H[comp]
                f1 = self.H[comp]
            else:
                # Generate fresh zero arrays for this component
                f0 = self.init_field_matrix(field, comp)
                f1 = self.init_field_matrix(field, comp)

            # Store either a tuple (prev, curr) or just curr array
            fields[comp] = f0 if not axis_type else (f0, f1)

            # Allocate time-series buffer if requested
            if num_frames:
                # Preallocate and set first frame to f0
                buf = bm.zeros((num_frames, *f0.shape), device=self.device)
                buf[0] = f0
                data[comp] = buf

        return fields, data

#################################### 激励源 #############################################        

    def set_source(self, position, source_type='sine', params=None, direction='z', time=None):
        """
        Register a new excitation source on the Yee grid.

        Args:
            position (tuple): (x, y, z) physical location of the source.
            source_type (str): 'sine', 'gaussian', 'ricker', or 'step'.
            params (dict, optional): Type-specific parameters. Defaults provided below.
            direction (str): 'x', 'y', or 'z' axis along which the source is applied.
            time (int or iterable, optional): Activation time step(s).

        Defaults:
            sine:     {'amplitude':1.0, 'frequency':None, 'PPW':15, 'phase':0.0}
            gaussian: {'amplitude':1.0, 'frequency':None, 'PPW':15, 'pulse_width':1.0, 'center_time':0.0}
            ricker:   {'amplitude':1.0, 'frequency':1.0, 'center_time':0.0}
            step:     {'amplitude':1.0, 'center_time':0}
        """
        # Map direction letter to axis index
        axis = 'xyz'.index(direction)

        # Default parameter sets per source type
        defaults = {
            'sine':     {'amplitude':1.0, 'frequency':None, 'PPW':15, 'phase':0.0},
            'gaussian': {'amplitude':1.0, 'frequency':None, 'PPW':15, 'pulse_width':1.0, 'center_time':0.0},
            'ricker':   {'amplitude':1.0, 'frequency':1.0, 'center_time':0.0},
            'step':     {'amplitude':1.0, 'center_time':0},
        }
        # Merge user params with defaults
        p = defaults.get(source_type, {}).copy()
        if params:
            p.update(params)

        self.sources.append({
            'type': source_type,
            'position': tuple(position),
            'axis': axis,
            'params': p,
            'times': set(time) if hasattr(time, '__iter__') else {time} if time is not None else None
        })

    def apply_sources(self, n, Ex=None, Ey=None, Ez=None):
        """
        At time step n, compute and inject all registered source values into the field arrays.

        Args:
            n (int): current time step.
            Ex, Ey, Ez (bm.ndarray): field component arrays.
        """
        t = n * self.dt

        inv_eta0 = 1.0 / self.eta0

        for src in self.sources:
            # Skip if not active at this time
            if src['times'] is not None and n not in src['times']:
                continue

            stype = src['type']
            A = src['params']['amplitude']
            axis = src['axis']

            # Compute base argument for sine-based oscillations
            if stype in ('sine', 'gaussian'):
                ppw = src['params']['PPW']
                f0  = src['params']['frequency']
                phase = src['params'].get('phase', 0.0)
                arg = bm.array((2 * bm.pi * (f0 * t if f0 else n * (self.R / ppw)) + phase),device=self.device)

            # Evaluate source_value by type
            if stype == 'sine':
                source_value = A * bm.sin(arg)

            elif stype == 'gaussian':
                tau = src['params']['pulse_width']
                t0  = src['params']['center_time']
                envelope = bm.exp(-((t - t0) / tau) ** 2)
                source_value = A * envelope * bm.sin(arg)

            elif stype == 'ricker':
                f0 = src['params']['frequency']
                t0 = src['params']['center_time']
                ξ = bm.array(bm.pi * f0 * (t - t0),device=self.device)
                source_value = A * (1 - 2 * ξ**2) * bm.exp(-ξ**2)

            elif stype == 'step':
                t0 = src['params']['center_time']
                source_value = A if n >= t0 else 0.0

            else:
                continue  # unknown type

            # Find grid index and inject
            idx = self.node_location(bm.asarray(src['position'], device=self.device))
            if self.dim == 2:
                Ez[idx] += source_value * inv_eta0
            else:
                i, j, k = idx
                # apply to whichever field component
                if axis == 0:
                    Ex[i, j, k] += source_value * inv_eta0
                elif axis == 1:
                    Ey[i, j, k] += source_value * inv_eta0
                else:
                    Ez[i, j, k] += source_value * inv_eta0

#################################### 设置物体 #############################################

    def set_object(self, position, permittivity=1.7**2, permeability=1,
               name=None):
        """
        Add an object to the FDTD simulation with specified material properties.

        Two modes for specifying `position`:
        1. Bounds list/tuple:
            - 2D: [x_start, x_end, y_start, y_end]
            - 3D: [x_start, x_end, y_start, y_end, z_start, z_end]
        2. Callable mask: function receiving coordinate arrays and returning boolean mask.

        Args:
            position (list|tuple|callable): Region specifier or mask function.
            permittivity (float): Relative permittivity (default=1.7**2).
            permeability (float): Relative permeability (default=1).
            name (str, optional): Identifier for the object.
        """
        # Helper to create grid points for bounds
        def _points_from_bounds(bounds):
            # bounds length 4 for 2D, length 6 for 3D
            dims = self.dim
            # generate per-dimension ranges
            coords = []
            for i in range(dims):
                start, end = bounds[2*i], bounds[2*i+1]
                coords.append(bm.arange(start, end, self.h, device=self.device))
            # meshgrid with 'ij' indexing
            mesh = bm.meshgrid(*coords, indexing='ij')
            # stack and reshape to (N, dim)
            stacked = bm.stack([m.ravel() for m in mesh], axis=1)
            return stacked

        # Determine point indices
        if callable(position):
            # all node coordinates
            node = self.mesh.node  # shape (N, dim)
            # apply mask function
            mask = position(*[node[:, i] for i in range(self.dim)])
            pos_indices = node[mask]
        else:
            arr = bm.asarray(position, device=self.device)
            if arr.ndim == 1 and arr.size == 2*self.dim:
                # interpret as bounds
                pos_indices = _points_from_bounds(arr.tolist())
            else:
                # already list of points
                pos_indices = arr

        # Store object metadata
        self.objects.append({
            'name': name,
            'position': pos_indices,
            'permittivity': permittivity,
            'permeability': permeability,
        })

    def apply_objects(self):
        """
        Update grid properties from stored objects.
        """
        for obj in self.objects:
            idx = self.node_location(obj['position'])
            self.permittivity[idx] = obj['permittivity']
            self.permeability[idx] = obj['permeability']

################################ 设置介电常数与磁导率 #####################################    
        
    def init_perm_matrix(self):
        """
        Convert permittivity and permeability to backend arrays,
        and if they come in as scalars, broadcast them to the full grid shape.
        """
        # Ensure arrays live on the correct device
        self.permittivity = bm.asarray(self.permittivity, device=self.device)
        self.permeability = bm.asarray(self.permeability, device=self.device)

        # If scalar, broadcast to the node-centered field grid
        node_shape = self.get_field_matrix("node").shape
        if self.permittivity.ndim == 0:
            self.permittivity = bm.full(node_shape, self.permittivity.item(), device=self.device)
        if self.permeability.ndim == 0:
            self.permeability = bm.full(node_shape, self.permeability.item(), device=self.device)

    def get_perm_matrix(self):
        """
        Compute staggered (half-step) permittivity and permeability arrays
        and store their inverses for the FDTD update equations.

        For 2D:
        - E-field lives on z, so permittivity_z stays at nodes.
        - H-field lives on x–y edges, so permeability is averaged along each axis.

        For 3D:
        - E-field has components along x, y, z faces → average permittivity on adjacent nodes.
        - H-field lives on edges → average permeability over the four surrounding nodes.
        """
        if self.dim == 2:
            # Magnetic permeability staggered at x- and y- edges
            μ = self.permeability
            μx = 0.5 * (μ[:, :-1] + μ[:, 1:])    # average in y-direction
            μy = 0.5 * (μ[:-1, :] + μ[1:, :])    # average in x-direction

            # Electric permittivity is node-centered (z-component is unchanged)
            εz = self.permittivity

            # Store inverses for update equations
            self.inverse_permeability_x = 1.0 / μx
            self.inverse_permeability_y = 1.0 / μy
            self.inverse_permittivity_z = 1.0 / εz

        else:
            # 3D: compute face-centered permittivity
            ε = self.permittivity
            # permittivity on faces perpendicular to x, y, z
            self.inverse_permittivity_x = 1.0 / (0.5 * (ε[:-1, :, :] + ε[1:, :, :]))
            self.inverse_permittivity_y = 1.0 / (0.5 * (ε[:, :-1, :] + ε[:, 1:, :]))
            self.inverse_permittivity_z = 1.0 / (0.5 * (ε[:, :, :-1] + ε[:, :, 1:]))

            # 3D: compute edge-centered permeability by averaging the four nodes around each edge
            μ = self.permeability
            μx = 0.25 * (
                μ[:, :-1, :-1] + μ[:, 1:, :-1] +
                μ[:, :-1, 1:] + μ[:, 1:, 1:]
            )
            μy = 0.25 * (
                μ[:-1, :, :-1] + μ[1:, :, :-1] +
                μ[:-1, :, 1:] + μ[1:, :, 1:]
            )
            μz = 0.25 * (
                μ[:-1, :-1, :] + μ[1:, -1:, :] +  # Note: [1:,:-1] and [:-1,1:] corrected for symmetry
                μ[:-1, 1:, :] + μ[1:, 1:, :]
            )

            # Store inverses for magnetic update
            self.inverse_permeability_x = 1.0 / μx
            self.inverse_permeability_y = 1.0 / μy
            self.inverse_permeability_z = 1.0 / μz

 #################################### 设置边界条件 #########################################

    def boundary(self,condition='PEC',m=6,ng=20):
        """
        Set boundary conditions
        """
        if condition=='PEC':
            self.boundary_condition='PEC'
        if condition=='PML' or condition=='UPML':
            self.boundary_condition='UPML'
            self.m=m
            self.ng=ng
        
 #################################### 更新函数 #############################################   

    def _register_update_methods(self):
        self._update_methods = {
            ('UPML', 2): self._update_2d_upml,
            ('UPML', 3): self._update_3d_upml,
            ('PEC',  2): self._update_2d_pec,
            ('PEC',  3): self._update_3d_pec,
        }

    def run(self, time=None, step=1):

        self.apply_objects()
        self.get_perm_matrix()
        time = self.NT if time is None else time
        try:
            fn = self._update_methods[(self.boundary_condition, self.dim)]
        except KeyError:
            raise ValueError(f"Unsupported BC/dim: {self.boundary_condition}/{self.dim}")
        fn(time, step)

    # ------- 公共辅助 -------

        ############################ UMPL边界条件 ############################## 
    def create_sigma_function(self, dim, ng, m, R0=0.001):
        """
        Create a sigma function to calculate the boundary values of a certain dimension.
        
        Arg:
        -Dim: Specify the calculation dimension (0 represents x, 1 represents y, 2 represents z).
        -Ng: The number of layers in the UPML layer
        -M: Index parameter.
                
        return:
        -A function f (p) that can calculate the sigma value of a specified dimension.
        """
        sup = bm.max(self.mesh.node, axis=0)
        inf = bm.min(self.mesh.node, axis=0)

        sigma = 1/ self.h

        # 预计算与指定维度相关的不变量
        pml_len = (sup[dim] - inf[dim]) * ng / (self.mesh.extent[(dim-1)*2+1] - self.mesh.extent[(dim-1)*2])

        l0 = inf[dim] + pml_len
        l1 = sup[dim] - pml_len
        def sigma_func(p):
            coord = p[..., dim]

            # 利用 bm.where 进行矢量化计算
            return bm.where(coord < l0, sigma * ((l0 - coord) / pml_len) ** m,
                            bm.where(coord > l1, sigma * ((coord - l1) / pml_len) ** m, 0))
        
        return sigma_func

    def _update_2d_upml(self,time=None,step = 1):

        R=self.R
        h=self.h
        m = self.m
        ng = self.ng

        sigma_x=self.create_sigma_function(0, ng, m)
        sigma_y=self.create_sigma_function(1, ng, m)

        sx0 = self.interpolation(sigma_x, intertype='node')
        sy0 = self.interpolation(sigma_y, intertype='node')

        sx1 = self.interpolation(sigma_x, intertype='edgex')
        sy1 = self.interpolation(sigma_y, intertype='edgex')

        sx2 = self.interpolation(sigma_x, intertype='edgey')
        sy2 = self.interpolation(sigma_y, intertype='edgey')
    
        c1 = (2 - sy2 * R * h) / (2 + sy2 * R * h)
        c2 = 2 * R / (2 + sy2 * R * h)

        c3 = (2 - sx1 * R * h) / (2 + sx1 * R * h)
        c4 = 2 * R / (2 + sx1 * R * h)

        c5 = (2 + sx2 * R * h) / 2
        c6 = (2 - sx2 * R * h) / 2

        c7 = (2 + sy1 * R * h) / 2
        c8 = (2 - sy1 * R * h) / 2

        c9 = (2 - sx0[1:-1, 1:-1] * R * h) / (2 + sx0[1:-1, 1:-1] * R * h)
        c10 = 2 * R / (2 + sx0[1:-1, 1:-1] * R * h)

        c11 = (2 - sy0[1:-1, 1:-1] * R * h) / (2 + sy0[1:-1, 1:-1] * R * h)
        c12 = 2 / (2 + sy0[1:-1, 1:-1] * R * h)

        for n in range(time+1):
            if n == 0:
                num=(time + 1)//step

                E_comps = ['z']
                H_comps = ['x', 'y']
                E, E_data = self._init_fields_dict('E', E_comps, num_frames=num,axis_type=None)
                D, _ = self._init_fields_dict('E', E_comps, num_frames=num)
                H, H_data = self._init_fields_dict('H', H_comps, num_frames=num,axis_type=None)
                B, _ = self._init_fields_dict('H', H_comps, num_frames=num)

    
            else:
                
                B['x'][1][:] = c1 * B['x'][0][:] - c2 *(E['z'][:, 1:] - E['z'][:, 0:-1])
                B['y'][1][:] = c3 * B['y'][0][:] + c4 *(E['z'][1:, :] - E['z'][0:-1, :])

                H['x'][:] += self.inverse_permeability_x * (c5 * B['x'][1][:] - c6 * B['x'][0][:])    
                H['y'][:] += self.inverse_permeability_y * (c7 * B['y'][1][:] - c8 * B['y'][0][:])

                D['z'][1][1:-1, 1:-1] = c9 * D['z'][0][1:-1, 1:-1] + c10 * (H['y'][1:, 1:-1] - H['y'][0:-1, 1:-1] - H['x'][1:-1, 1:] + H['x'][1:-1, 0:-1])
                
                E['z'][1:-1, 1:-1] = c11 * E['z'][1:-1, 1:-1] + c12 * self.inverse_permittivity_z[1:-1, 1:-1] * (D['z'][1][1:-1, 1:-1] - D['z'][0][1:-1, 1:-1])

                self.apply_sources(n,Ez=E['z'])
                
                B['x'][0][:] = B['x'][1][:]
                B['y'][0][:] = B['y'][1][:]
                D['z'][0][:] = D['z'][1][:]

                if n % step == 0:
                    idx=n//step
                    E_data['z'][idx] = E['z']
                    H_data['x'][idx] = H['x']
                    H_data['y'][idx] = H['y']


        self.E['z'] = E_data['z']*self.eta0
        self.H['x'] = H_data['x']
        self.H['y'] = H_data['y']


    def _update_3d_upml(self,time,step):

        m=self.m
        ng=self.ng
        R=self.R
        h=self.h
        
        sigma_x=self.create_sigma_function(0, ng, m)
        sigma_y=self.create_sigma_function(1, ng, m)
        sigma_z=self.create_sigma_function(2, ng, m)
        sx0 = self.interpolation(sigma_x,intertype='facex')
        sy0 = self.interpolation(sigma_y,intertype='facex')
        sz0 = self.interpolation(sigma_z,intertype='facex')

        sx1 = self.interpolation(sigma_x,intertype='facey')
        sy1 = self.interpolation(sigma_y,intertype='facey')
        sz1 = self.interpolation(sigma_z,intertype='facey')

        sx2 = self.interpolation(sigma_x,intertype='facez')
        sy2 = self.interpolation(sigma_y,intertype='facez')
        sz2 = self.interpolation(sigma_z,intertype='facez')

        sx3 = self.interpolation(sigma_x,intertype='edgex')
        sy3 = self.interpolation(sigma_y,intertype='edgex')
        sz3 = self.interpolation(sigma_z,intertype='edgex')

        sx4 = self.interpolation(sigma_x,intertype='edgey')
        sy4 = self.interpolation(sigma_y,intertype='edgey')
        sz4 = self.interpolation(sigma_z,intertype='edgey')

        sx5 = self.interpolation(sigma_x,intertype='edgez')
        sy5 = self.interpolation(sigma_y,intertype='edgez')
        sz5 = self.interpolation(sigma_z,intertype='edgez')
    
        c1 = (2 - sz0 * R * h) / (2 + sz0 * R * h)
        
        c2 = 2 * R / (2 + sz0 * R * h)

        c3 = (2 - sx1 * R * h) / (2 + sx1 * R * h)
        c4 = 2 * R / (2 + sx1 * R * h)

        c5 = (2 - sy2 * R * h) / (2 + sy2 * R * h)
        c6 = 2 * R / (2 + sy2 * R * h)

        c7 = (2 - sy0 * R * h) / (2 + sy0 * R * h)
        c8 = (2 + sx0 * R * h) / (2 + sy0 * R * h)
        c9 = (2 - sx0 * R * h) / (2 + sy0 * R * h)

        c10 = (2 - sz1 * R * h) / (2 + sz1 * R * h)
        c11 = (2 + sy1 * R * h) / (2 + sz1 * R * h)
        c12 = (2 - sy1 * R * h) / (2 + sz1 * R * h)

        c13 = (2 - sx2 * R * h) / (2 + sx2 * R * h)
        c14 = (2 + sz2 * R * h) / (2 + sx2 * R * h)
        c15 = (2 - sz2 * R * h) / (2 + sx2 * R * h)

        c16 = (2 - sz3[:, 1:-1, 1:-1] * R * h) / (2 + sz3[:, 1:-1, 1:-1] * R * h)
        c17 = 2 * R / (2 + sz3[:, 1:-1, 1:-1] * R * h)

        c18 = (2 - sx4[1:-1, :, 1:-1] * R * h) / (2 + sx4[1:-1, :, 1:-1] * R * h)
        c19 = 2 * R / (2 + sx4[1:-1, :, 1:-1] * R * h)

        c20 = (2 - sy5[1:-1, 1:-1, :] * R * h) / (2 + sy5[1:-1, 1:-1, :] * R * h)
        c21 = 2 * R / (2 + sy5[1:-1, 1:-1, :] * R * h)

        c22 = (2 - sy3[:, 1:-1, 1:-1] * R * h) / (2 + sy3[:, 1:-1, 1:-1] * R * h)
        c23 = (2 + sx3[:, 1:-1, 1:-1] * R * h) / (2 + sy3[:, 1:-1, 1:-1] * R * h)
        c24 = (2 - sx3[:, 1:-1, 1:-1] * R * h) / (2 + sy3[:, 1:-1, 1:-1] * R * h)

        c25 = (2 - sz4[1:-1, :, 1:-1] * R * h) / (2 + sz4[1:-1, :, 1:-1] * R * h)
        c26 = (2 + sy4[1:-1, :, 1:-1] * R * h) / (2 + sz4[1:-1, :, 1:-1] * R * h)
        c27 = (2 - sy4[1:-1, :, 1:-1] * R * h) / (2 + sz4[1:-1, :, 1:-1] * R * h)

        c28 = (2 - sx5[1:-1, 1:-1, :] * R * h) / (2 + sx5[1:-1, 1:-1, :] * R * h)
        c29 = (2 + sz5[1:-1, 1:-1, :] * R * h) / (2 + sx5[1:-1, 1:-1, :] * R * h)
        c30 = (2 - sz5[1:-1, 1:-1, :] * R * h) / (2 + sx5[1:-1, 1:-1, :] * R * h)

        for n in range(time+1):
            
            if n == 0:
                num=(time + 1)//step
                comps = ['x', 'y', 'z']

                E, E_data = self._init_fields_dict('E', comps, num_frames=num)
                D, _      = self._init_fields_dict('E', comps, num_frames=None)
                H, H_data = self._init_fields_dict('H', comps, num_frames=num)
                B, _      = self._init_fields_dict('H', comps, num_frames=None)

            else:
                B['x'][1][:] = c1*B['x'][0] - c2*((E['z'][0][:,1:,:] - E['z'][0][:,:-1,:]) - (E['y'][0][:,:,1:] - E['y'][0][:,:,:-1]))
                B['y'][1][:] = c3*B['y'][0] - c4*((E['x'][0][:,:,1:] - E['x'][0][:,:,:-1]) - (E['z'][0][1:,:,:] - E['z'][0][:-1,:,:]))
                B['z'][1][:] = c5*B['z'][0] - c6*((E['y'][0][1:,:,:] - E['y'][0][:-1,:,:]) - (E['x'][0][:,1:,:] - E['x'][0][:,:-1,:]))
                # H 更新
                H['x'][1][:] = c7*H['x'][0] + self.inverse_permeability_x * (c8*B['x'][1] - c9*B['x'][0])
                H['y'][1][:] = c10*H['y'][0] + self.inverse_permeability_y * (c11*B['y'][1] - c12*B['y'][0])
                H['z'][1][:] = c13*H['z'][0] + self.inverse_permeability_z * (c14*B['z'][1] - c15*B['z'][0])
                # D 更新
                D['x'][1][:,1:-1,1:-1] = c16*D['x'][0][:,1:-1,1:-1] + c17 * ((H['z'][1][:,1:,1:-1] - H['z'][1][:,:-1,1:-1]) - (H['y'][1][:,1:-1,1:]-H['y'][1][:,1:-1,:-1]))
                D['y'][1][1:-1,:,1:-1] = c18*D['y'][0][1:-1,:,1:-1] + c19 * ((H['x'][1][1:-1,:,1:] - H['x'][1][1:-1,:,:-1]) - (H['z'][1][1:,:,1:-1]-H['z'][1][:-1,:,1:-1]))
                D['z'][1][1:-1,1:-1,:] = c20*D['z'][0][1:-1,1:-1,:] + c21 * ((H['y'][1][1:,1:-1,:] - H['y'][1][:-1,1:-1,:]) - (H['x'][1][1:-1,1:,:]-H['x'][1][1:-1,:-1,:]))
                # E 更新
                E['x'][1][:,1:-1,1:-1] = c22*E['x'][0][:,1:-1,1:-1] + self.inverse_permittivity_x[:,1:-1,1:-1] * (c23*D['x'][1][:,1:-1,1:-1] - c24*D['x'][0][:,1:-1,1:-1])
                E['y'][1][1:-1,:,1:-1] = c25*E['y'][0][1:-1,:,1:-1] + self.inverse_permittivity_y[1:-1,:,1:-1] * (c26*D['y'][1][1:-1,:,1:-1] - c27*D['y'][0][1:-1,:,1:-1])
                E['z'][1][1:-1,1:-1,:] = c28*E['z'][0][1:-1,1:-1,:] + self.inverse_permittivity_z[1:-1,1:-1,:] * (c29*D['z'][1][1:-1,1:-1,:] - c30*D['z'][0][1:-1,1:-1,:])
                # 应用源
                self.apply_sources(n, E['x'][1], E['y'][1], E['z'][1])

                for c in comps:
                    E[c][0][:] = E[c][1][:]
                    H[c][0][:] = H[c][1][:]
                    D[c][0][:] = D[c][1][:]
                    B[c][0][:] = B[c][1][:]
                if n%step==0:
                    idx=n//step
                    for c in comps:
                        E_data[c][idx]=E[c][1]
                        H_data[c][idx]=H[c][1]

                print('t={}'.format(n))
        for c in comps:
            self.E[c]=E_data[c]*self.eta0
            self.H[c]=H_data[c]
        


#################################### PEC边界条件 #############################################

    def _update_2d_pec(self,time=None,step = 1):
        R=self.R

        for n in range(time+1):

            if n == 0:
                num=(time + 1)//step
                E_comps = ['z']
                H_comps = ['x', 'y']
                E, E_data = self._init_fields_dict('E', E_comps, num_frames=num,axis_type=None)
                H, H_data = self._init_fields_dict('H', H_comps, num_frames=num,axis_type=None)
            else:

                H['x'] -= R *self.inverse_permeability_x *(E['z'][:, 1:] - E['z'][:, 0:-1])
                H['y'] += R *self.inverse_permeability_y *(E['z'][1:, :] - E['z'][0:-1, :])
  
                E['z'][1:-1, 1:-1] += R * self.inverse_permittivity_z[1:-1, 1:-1]*(H['y'][1:, 1:-1] - H['y'][0:-1, 1:-1] - H['x'][1:-1, 1:] + H['x'][1:-1, 0:-1])

                self.apply_sources(n,Ez=E['z'])

                if n % step == 0:
                    idx=n//step
                    E_data['z'][idx] = E['z']
                    H_data['x'][idx] = H['x']
                    H_data['y'][idx] = H['y']


        self.E['z'] = E_data['z']*self.eta0
        self.H['x'] = H_data['x']
        self.H['y'] = H_data['y']

    def _update_3d_pec(self,time=None, step=1):

        R=self.R

        for n in range(time+1):
            if n == 0:
                num = (time + 1)//step
                comps = ['x', 'y', 'z']

                E, E_data = self._init_fields_dict('E', comps, num_frames=num,axis_type=None)
                H, H_data = self._init_fields_dict('H', comps, num_frames=num,axis_type=None)

            else:
                H['x'] -= R * self.inverse_permeability_x * (
                    (E['z'][:, 1:, :] - E['z'][:, :-1, :]) - (E['y'][:, :, 1:] - E['y'][:, :, :-1]))
                
                H['y'] -= R * self.inverse_permeability_y * (
                    (E['x'][:, :, 1:] - E['x'][:, :, :-1]) - (E['z'][1:, :, :] - E['z'][:-1, :, :]))
                
                H['z'] -= R * self.inverse_permeability_z * (
                    (E['y'][1:, :, :] - E['y'][:-1, :, :]) - (E['x'][:, 1:, :] - E['x'][:, :-1, :]))

                E['x'][:, 1:-1, 1:-1] += R * self.inverse_permittivity_x[:, 1:-1, 1:-1] * (
                    (H['z'][:, 1:, 1:-1] - H['z'][:, :-1, 1:-1])
                - (H['y'][:, 1:-1, 1:] - H['y'][:, 1:-1, :-1])
                )

                E['y'][1:-1, :, 1:-1] += R * self.inverse_permittivity_y[1:-1, :, 1:-1] * (
                    (H['x'][1:-1, :, 1:] - H['x'][1:-1, :, :-1])
                - (H['z'][1:, :, 1:-1] - H['z'][:-1, :, 1:-1])
                )

                E['z'][1:-1, 1:-1, :] += R * self.inverse_permittivity_z[1:-1, 1:-1, :] * (
                    (H['y'][1:, 1:-1, :] - H['y'][:-1, 1:-1, :])
                - (H['x'][1:-1, 1:, :] - H['x'][1:-1, :-1, :])
                )

                self.apply_sources(n,E['x'],E['y'],E['z'])

                if n%step ==0 :
                    idx = n//step
                    for c in comps:
                        E_data[c][idx] = E[c]
                        H_data[c][idx] = H[c]

        for c in comps:
            self.E[c] = E_data[c]*self.eta0
            self.H[c] = H_data[c]  

#################################### 可视化 #############################################
 
    def show_animation(self,
                   fig: Figure,
                   axes: Union[Axes, Axes3D],
                   box: List[float],
                   data_array: bm.array,
                   fname: str = 'test.mp4',
                   slice_plane: Optional[Tuple] = None,
                   frames: int = 1000,
                   interval: int = 50,
                   plot_type: str = 'imshow',
                   cmap: str = 'rainbow',
                   step: int = 0) -> None:
        """
        Generate animation and save it
        """
        import matplotlib.animation as animation
        from matplotlib.patches import Rectangle
        import numpy as np

        
        def reset_axes(axes: Union[Axes, Axes3D], center=True, margin=0.05):

            axes.clear()
            axes.set_aspect('auto')
            axes.autoscale()
            if center:
                # Set even margins
                axes.margins(margin)
        # Cleaning: empty the main shaft and multiple sub shafts
        reset_axes(axes)
        for ax in fig.axes[1:]:
            fig.delaxes(ax)

        # Determine data dimension
        dim = data_array.ndim
        is_1d = (dim == 1)
        mesh = getattr(self, 'mesh', None)

        def apply_slice(data, plane):
            if plane is None or data.ndim != 3:
                return data
            axis, idx = plane
            return {
                'x': data[idx, :, :],
                'y': data[:, idx, :],
                'z': data[:, :, idx]
            }[axis]

        # Initializing drawing objects
        if is_1d:
            # One dimensional time series polyline (progressive drawing)
            x = np.arange(frames)
            y = np.full(frames, np.nan, dtype=float)
            line, = axes.plot(x, y, lw=2)
            axes.set_xlim(0, frames - 1)
            mn, mx = float(np.nanmin(data_array)), float(np.nanmax(data_array))
            axes.set_ylim(mn, mx)
            axes.set_xlabel('nt')
            axes.set_ylabel('field')
        else:
            # 2D/3D visualization
            first = apply_slice(data_array[0], slice_plane)
            if isinstance(axes, Axes3D) and plot_type == 'surface':
                X, Y = mesh.node[..., 0], mesh.node[..., 1]
                surf = axes.plot_surface(
                    X, Y, first, cmap=cmap,
                    vmin=box[4], vmax=box[5], rstride=1, cstride=1)
                axes.set(xlim=(box[0], box[1]), ylim=(box[2], box[3]), zlim=(box[4], box[5]))
                fig.colorbar(surf, ax=axes, shrink=0.5, aspect=10)
            else:
                surf = axes.imshow(
                    first, cmap=cmap,
                    vmin=box[4], vmax=box[5],
                    extent=box[:4], origin='lower', interpolation='none')
                axes.invert_yaxis()
                axes.set(xlabel='y', ylabel='x')
                fig.colorbar(surf, ax=axes)

            # Mark objects (2D)
            if getattr(self, 'dim', None) == 2 and getattr(self, 'objects', None):
                for obj in self.objects:
                    pos = bm.array(obj['position'], device=self.device)
                    if pos.size == 0:
                        continue
                    x_min, y_min = pos.min(axis=0)
                    x_max, y_max = pos.max(axis=0)
                    patch = Rectangle(
                        (y_min, x_max), width=y_max - y_min, height=-(x_max - x_min),
                        linewidth=0, edgecolor='none', facecolor=(1, 0, 0, 0.2))
                    axes.add_patch(patch)

            # Mark UPML boundaries
            if getattr(self, 'boundary_condition', '').upper() == 'UPML':
                x_len = box[1] - box[0]
                y_len = box[3] - box[2]
                upml = self.ng * self.h
                for x0, y0, w, h in [
                    (box[0], box[2], upml, y_len),
                    (box[1] - upml, box[2], upml, y_len),
                    (box[0], box[2], x_len, upml),
                    (box[0], box[3] - upml, x_len, upml)
                ]:
                    axes.add_patch(Rectangle((x0, y0), w, h, linewidth=0,
                                            edgecolor='none', facecolor=(0, 0, 1, 0.1)))

        # Update function
        def update(n):
            title = f"nt={n:05d}"
            axes.set_title(title)
            print(title)

            if is_1d:
                # Draw lines in progress
                y[n] = data_array[n]
                line.set_ydata(y)
                artists_out = (line,)
            else:
                frame = apply_slice(data_array[n], slice_plane)
                if isinstance(axes, Axes3D) and plot_type == 'surface':
                    axes.collections.clear()
                    new_surf = axes.plot_surface(
                        X, Y, frame, cmap=cmap,
                        vmin=box[4], vmax=box[5], rstride=1, cstride=1)
                    artists_out = (new_surf,)
                else:
                    surf.set_array(frame)
                    artists_out = (surf,)

            if step and (n % step == 0):
                fig.savefig(f"{fname[:-4]}_{n:05d}.png")
            return artists_out

        # Create animation and save
        ani = animation.FuncAnimation(
            fig, update, frames=frames,
            interval=interval, blit=is_1d
        )
        ani.save(f"{fname}")

