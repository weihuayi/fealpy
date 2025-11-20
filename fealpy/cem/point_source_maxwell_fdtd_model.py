from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.model import ComputationalModel
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from typing import Union, List, Optional, Tuple
import numpy as np
import math
import warnings

# Physical constants
c0 = 299792458.0
mu0 = 4.0 * math.pi * 1e-7
eta0: float = mu0 * c0


class PointSourceMaxwellFDTDModel(ComputationalModel):
    """FDTD model for Maxwell's equations with point sources."""
    
    def __init__(self, pde, n, options=None):
        """Initialize FDTD model.
        
        Args:
            pde: PointSourceMaxwell instance
            n: Grid resolution
            options: Configuration options
        """
        super().__init__(pbar_log=True, log_level="INFO")
        self.pde = pde
        self.options = options or {}
        
        # Initialize mesh
        domain = pde.domain
        from .mesh.yee_uniform_mesher import YeeUniformMesher
        
        # Determine grid size based on dimension
        if pde.geo_dimension == 2:
            self.mesh = YeeUniformMesher(domain=domain, nx=n, ny=n, nz=0)
        else:
            self.mesh = YeeUniformMesher(domain=domain, nx=n, ny=n, nz=n)
        
        # Create material arrays with object information
        eps_array, mu_array = self._create_material_arrays()
        
        # Initialize FDTD solver with material arrays
        self._init_fdtd_solver(eps_array, mu_array)
        
        # Initialize source manager
        self._init_source_manager()
        
        # Initialize field storage
        self._init_fields()
        
        # Apply configuration options
        self._apply_options()
        
        # Time state
        self.current_step = 0
        self.current_time = 0.0

    def _create_material_arrays(self):
        """Create permittivity and permeability arrays with object information."""
        # Get node grid shape
        node_shape = self.mesh.get_field_matrix("node").shape
        
        # Initialize with background material
        eps_array = bm.full(node_shape, self.pde.eps, dtype=bm.float64)
        mu_array = bm.full(node_shape, self.pde.mu, dtype=bm.float64)
        
        # Apply object material parameters
        objects = self.pde.get_object_config()
        for obj in objects:
            self._apply_object_materials(obj, eps_array, mu_array)
        
        self.logger.info(f"Created material arrays with {len(objects)} objects")
        return eps_array, mu_array

    def _apply_object_materials(self, obj, eps_array, mu_array):
        """Apply single object's material parameters to arrays."""
        box = obj['box']
        
        # Convert physical coordinates to grid indices
        if self.mesh.TD == 2:
            xmin_idx, xmax_idx = self._physical_to_index(box[0], box[1], 'x')
            ymin_idx, ymax_idx = self._physical_to_index(box[2], box[3], 'y')
            
            # Update material parameters - fix index order issue
            if obj.get('eps') is not None:
                eps_array[ymin_idx:ymax_idx, xmin_idx:xmax_idx] = obj['eps']
            if obj.get('mu') is not None:
                mu_array[ymin_idx:ymax_idx, xmin_idx:xmax_idx] = obj['mu']
        else:
            xmin_idx, xmax_idx = self._physical_to_index(box[0], box[1], 'x')
            ymin_idx, ymax_idx = self._physical_to_index(box[2], box[3], 'y')
            zmin_idx, zmax_idx = self._physical_to_index(box[4], box[5], 'z')
            
            # Update material parameters - fix index order issue
            if obj.get('eps') is not None:
                eps_array[zmin_idx:zmax_idx, ymin_idx:ymax_idx, xmin_idx:xmax_idx] = obj['eps']
            if obj.get('mu') is not None:
                mu_array[zmin_idx:zmax_idx, ymin_idx:ymax_idx, xmin_idx:xmax_idx] = obj['mu']

    def _physical_to_index(self, phys_min, phys_max, axis):
        """Convert physical coordinate range to grid index range."""
        domain = self.pde.domain
        if axis == 'x':
            domain_min, domain_max = domain[0], domain[1]
            n = self.mesh.nx
        elif axis == 'y':
            domain_min, domain_max = domain[2], domain[3]
            n = self.mesh.ny
        else:  # 'z'
            domain_min, domain_max = domain[4], domain[5]
            n = self.mesh.nz
        
        idx_min = int((phys_min - domain_min) / (domain_max - domain_min) * n)
        idx_max = int((phys_max - domain_min) / (domain_max - domain_min) * n)
        
        # Ensure indices are within valid range
        idx_min = max(0, min(idx_min, n-1))
        idx_max = max(0, min(idx_max, n))
        
        return idx_min, idx_max

    def _init_fdtd_solver(self, eps_array, mu_array):
        """Initialize FDTD solver with material arrays."""
        from .simulation.step_fdtd_maxwell import StepFDTDMaxwell
        
        dt = self.options.get('dt')
        R = self.options.get('R')
        boundary = self.options.get('boundary', 'PEC')
        pml_width = self.options.get('pml_width', 8)
        pml_m = self.options.get('pml_m', 5.0)
        
        self.fdtd = StepFDTDMaxwell(
            yee=self.mesh,
            dt=dt,
            R=R,
            boundary=boundary,
            pml_width=pml_width,
            pml_m=pml_m,
            eps=eps_array,  # Pass material arrays with object information
            mu=mu_array     # Pass material arrays with object information
        )
        
        self.logger.info(f"FDTD parameters: R={self.fdtd.R:.4f}, dt={self.fdtd.dt:.2e}s")

    def _init_source_manager(self):
        """Initialize source manager from PDE configuration."""
        from .model.source import SourceManager, Source
        
        self.source_manager = SourceManager()
        
        # Convert PDE source configurations to Source objects
        for src_cfg in self.pde.get_source_config():
            waveform_name = src_cfg['waveform']
            waveform_params = src_cfg['waveform_params']
            
            # Convert waveform name to callable function
            waveform_func = self._get_waveform_function(waveform_name, waveform_params)
            
            source = Source(
                position=src_cfg['position'],
                comp=src_cfg['comp'],
                waveform=waveform_func,
                injection=src_cfg['injection'],
                amplitude=src_cfg['amplitude'],
                spread=src_cfg['spread']
            )
            self.source_manager.add(source)

        self.logger.info(f"Initialized {len(self.source_manager.sources)} sources")

    def _get_waveform_function(self, name, params):
        """Convert waveform name and parameters to callable function."""
        from .model.source import (
            gaussian_pulse, ricker_wavelet, 
            sinusoid, gaussian_enveloped_sine
        )
        
        waveforms = {
            'gaussian': gaussian_pulse,
            'ricker': ricker_wavelet, 
            'sinusoid': sinusoid,
            'gaussian_enveloped_sine': gaussian_enveloped_sine
        }
        
        if name not in waveforms:
            raise ValueError(f"Unknown waveform: {name}")
        
        # Create partial function with fixed parameters
        import functools
        return functools.partial(waveforms[name], **params)

    def _init_fields(self):
        """Initialize electromagnetic fields."""
        # Determine required field components based on dimension
        if self.mesh.TD == 2:
            # 2D TM mode: Ez, Hx, Hy
            self.E = {'x': None, 'y': None, 'z': self.mesh.init_field_matrix('E', 'z')}
            self.H = {'x': self.mesh.init_field_matrix('H', 'x'), 
                     'y': self.mesh.init_field_matrix('H', 'y'), 
                     'z': None}
        else:
            # 3D: All components
            self.E = {comp: self.mesh.init_field_matrix('E', comp) for comp in 'xyz'}
            self.H = {comp: self.mesh.init_field_matrix('H', comp) for comp in 'xyz'}
        
        # For UPML boundary conditions, auxiliary fields are needed
        if self.fdtd.boundary == 'UPML':
            self.D = {comp: bm.zeros_like(self.E[comp]) if self.E[comp] is not None else None 
                     for comp in 'xyz'}
            self.B = {comp: bm.zeros_like(self.H[comp]) if self.H[comp] is not None else None 
                     for comp in 'xyz'}
        else:
            self.D = self.B = None

    def _apply_options(self):
        """Apply configuration options."""
        if self.options:
            # Set run parameters
            self.maxstep = self.options.get('maxstep', 1000)
            self.save_every = self.options.get('save_every', 10)

    def _normalize_initial_fields(self):
        """Convert initial fields to dimensionless form."""
        # Normalize electric field by η₀
        for comp in 'xyz':
            if self.E[comp] is not None:
                self.E[comp] = self.E[comp] / eta0
        
        # Also normalize D field if it exists
        if self.D is not None:
            for comp in 'xyz':
                if self.D[comp] is not None:
                    self.D[comp] = self.D[comp] / eta0

    def _denormalize_final_fields(self, field_history):
        """Denormalize final field data back to physical dimensions."""
        # Denormalize current E and D field components
        for comp in 'xyz':
            if self.E[comp] is not None:
                self.E[comp] = self.E[comp] * eta0
            if self.D is not None and self.D[comp] is not None:
                self.D[comp] = self.D[comp] * eta0
        
        # Denormalize all E components in field history
        for snapshot in field_history:
            for comp in 'xyz':
                if snapshot['E'][comp] is not None:
                    snapshot['E'][comp] = snapshot['E'][comp] * eta0
    
    

    @variantmethod('main')
    def run(self, nt=None, save_every=None):
        """Main run loop."""
        self.run_str = "main"
        nt = nt or self.maxstep
        
        save_every = save_every or self.save_every

        self._normalize_initial_fields()
        # Initialize field history
        field_history = []

        for step in range(nt):
            # Save field snapshot
            if step % save_every == 0:
                self._save_field_snapshot(field_history, step, self.current_time)
            
            # Single step update
            self.run_one_step()
            
            # Log progress
            if step % 100 == 0:
                self.logger.info(f"Step {step}/{nt}, t={self.current_time:.3e}s")
        
        self._denormalize_final_fields(field_history)

        return field_history

    @variantmethod('one_step')
    def run_one_step(self):
        """Single step execution."""
        self.run_str = "one_step"
        
        # Apply source terms
        self.source_manager.apply_all(self.current_time, self.mesh, self.E, self.H)
        
        # FDTD update
        if self.fdtd.boundary == 'UPML':
            self.E, self.H, self.D, self.B = self.fdtd.update(
                self.E, self.H, self.D, self.B)
        else:
            self.E, self.H = self.fdtd.update(self.E, self.H)

        # Update time state
        self.current_step += 1
        self.current_time = self.current_step * self.fdtd.dt
        
        return self.E, self.H
    
    def _save_field_snapshot(self, history, step, t):
        """Save field snapshot."""
        snapshot = {
            'step': step,
            'time': t,
            'E': {k: bm.array(v) if v is not None else None for k, v in self.E.items()},
            'H': {k: bm.array(v) if v is not None else None for k, v in self.H.items()}
        }
        
        history.append(snapshot)

    def show_field(self,
                field_history: List,
                step_index: Optional[int] = None,
                field_component: str = 'Ez',
                fig: Optional[Figure] = None,
                axes: Optional[Union[Axes, Axes3D]] = None,
                slice_plane: Optional[Tuple] = None,
                plot_type: str = 'imshow',
                cmap: str = 'viridis') -> Tuple[Figure, Union[Axes, Axes3D]]:
        """
        Display one snapshot from field_history.

        Fixed behaviors:
        - Safe handling of step_index (None -> last frame, bounds checked)
        - Consistent `extent` ordering: [xmin, xmax, ymin, ymax]
        - Avoid unnecessary invert_yaxis() calls; use origin='lower'
        - Clearer axis labels (X,Y correspond to physical coords)
        """
        
        # Validation
        if not field_history:
            raise ValueError("field_history is empty")

        if step_index is None:
            step_index = len(field_history) - 1
        if not (0 <= step_index < len(field_history)):
            raise ValueError(f"step_index {step_index} out of range [0, {len(field_history)-1}]")

        snapshot = field_history[step_index]

        # Prepare figure/axes
        if fig is None or axes is None:
            if plot_type == 'surface' and self.mesh.TD == 3:
                fig = plt.figure(figsize=(10, 8))
                axes = fig.add_subplot(111, projection='3d')
            else:
                fig, axes = plt.figure(figsize=(10, 8)), plt.subplot(111)

        # Reset axes
        self._reset_axes(axes)

        # Extract field array
        key = field_component[-1].lower()
        if field_component.startswith('E'):
            field_data = snapshot['E'].get(key)
        else:
            field_data = snapshot['H'].get(key)

        if field_data is None:
            raise ValueError(f"Field component {field_component} is not available")

        # Convert to numpy (boundary between backend and visualization)
        if hasattr(field_data, 'numpy'):
            data = field_data.numpy()
        else:
            data = bm.to_numpy(field_data)

        # For 3D data, default slice is mid-z if not provided
        if data.ndim == 3 and slice_plane is None:
            slice_plane = ('z', data.shape[2] // 2)

        data = self._apply_slice(data, slice_plane)

        # Construct consistent box: [xmin, xmax, ymin, ymax, vmin, vmax]
        domain = getattr(self.pde, 'domain', None)
        if data.ndim == 2 and domain is not None and len(domain) >= 4:
            xmin, xmax = float(domain[0]), float(domain[1])
            ymin, ymax = float(domain[2]), float(domain[3])
        else:
            # Fallback to index-based coordinates
            xmin, xmax = 0.0, float(data.shape[1] - 1)
            ymin, ymax = 0.0, float(data.shape[0] - 1)

        vmin = float(np.nanmin(data)) if np.isfinite(np.nanmin(data)) else 0.0
        vmax = float(np.nanmax(data)) if np.isfinite(np.nanmax(data)) else 1.0
        box = [xmin, xmax, ymin, ymax, vmin, vmax]

        # Plot
        if plot_type == 'surface' and isinstance(axes, Axes3D) and data.ndim == 2:
            X, Y = self._create_mesh_grid_for_slice(data.shape, slice_plane)
            surf = axes.plot_surface(X, Y, data, cmap=cmap, vmin=box[4], vmax=box[5], rstride=1, cstride=1)
            axes.set_xlim(box[0], box[1])
            axes.set_ylim(box[2], box[3])
            axes.set_zlim(box[4], box[5])
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            axes.set_zlabel(field_component)
            fig.colorbar(surf, ax=axes, shrink=0.6)
        else:
            # extent expects [xmin, xmax, ymin, ymax]
            im = axes.imshow(data, cmap=cmap, vmin=box[4], vmax=box[5], extent=box[:4], origin='lower', interpolation='nearest', aspect='auto')
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            cbar = fig.colorbar(im, ax=axes)
            cbar.set_label(field_component)

        # Mark UPML
        self._mark_upml_boundaries(axes, box)

        # Title
        slice_info = f" at {slice_plane}" if slice_plane else ""
        axes.set_title(f"{field_component}{slice_info} at t={snapshot['time']:.2e}s (step {snapshot['step']})")

        return fig, axes

    def plot_time_series(self,
                     field_history: List,
                     positions: List[Tuple[float, ...]],
                     field_component: str = 'Ez',
                     fig: Optional[Figure] = None,
                     axes: Optional[Axes] = None,
                     interpolate: bool = False) -> Tuple[Figure, Axes]:
        """
        Plot time series at specified physical positions (or grid positions) based on field_history.

        Args:
            positions: List of tuples, physical coordinates or node coordinates, 
                      length should match self.mesh.TD
            interpolate: If True, attempt bilinear/trilinear interpolation for float indices
                       (currently implemented as nearest neighbor, with extension points reserved)
        """
        if not field_history:
            raise ValueError("field_history is empty, cannot plot time series")

        # Prepare figure
        if fig is None or axes is None:
            fig, axes = plt.subplots(figsize=(12, 6))

        self._reset_axes(axes)

        # Time axis
        time_data = [snapshot.get('time', i) for i, snapshot in enumerate(field_history)]
        n_steps = len(field_history)
        n_pos = len(positions)

        # Preallocate result array (float32 to save memory)
        field_data = np.full((n_pos, n_steps), np.nan, dtype=np.float32)

        # Pre-map positions to indices (avoid calling mesh.node_location every step)
        idx_list = []
        for pos in positions:
            try:
                # Try direct mapping via mesh.node_location once
                idx = self.mesh.node_location(pos)
                # node_location may return (i,j) or (i,j,k) or float, unify as tuple of ints or floats
                if isinstance(idx, (list, tuple)):
                    idx_list.append(tuple(idx))
                else:
                    # Single value return or exception, mark as None
                    idx_list.append(None)
            except Exception:
                idx_list.append(None)

        # Helper: convert an index-like (possibly float) to integer grid indices and flag if float
        def _idx_to_ints(idx_like, shape):
            """
            idx_like: tuple of ints/floats or None
            shape: array shape
            Returns (is_valid, is_float, ints_tuple)
            """
            if idx_like is None:
                return False, False, None
            try:
                # Ensure tuple
                if not isinstance(idx_like, (list, tuple)):
                    return False, False, None
                # Check length matches
                if len(idx_like) != self.mesh.TD:
                    return False, False, None
                ints = []
                is_float = False
                for a, dim in zip(idx_like, shape):
                    if a is None:
                        return False, False, None
                    if isinstance(a, float):
                        is_float = True
                        ai = int(round(a))
                    else:
                        ai = int(a)
                    # Clamp
                    ai = max(0, min(dim - 1, ai))
                    ints.append(ai)
                return True, is_float, tuple(ints)
            except Exception:
                return False, False, None

        # Main loop: convert array to numpy once per step, then extract values for all positions
        for step, snapshot in enumerate(field_history):
            # Get field array (backend object)
            if field_component.startswith('E'):
                field_array = snapshot['E'].get(field_component[-1].lower())
            else:
                field_array = snapshot['H'].get(field_component[-1].lower())

            if field_array is None:
                # Current snapshot doesn't have this component
                continue

            # Convert field to numpy (once per step)
            try:
                if hasattr(field_array, 'numpy'):
                    arr = field_array.numpy()
                else:
                    arr = bm.to_numpy(field_array)
            except Exception as e:
                warnings.warn(f"Failed to convert field data to numpy (step {step}): {e}")
                continue

            # Ensure arr is numpy ndarray
            if not isinstance(arr, np.ndarray):
                arr = np.array(arr, dtype=np.float32)

            # Extract values for each position (using precomputed idx_list)
            for i_pos, pos in enumerate(positions):
                idx_like = idx_list[i_pos]
                ok, is_float, ints = _idx_to_ints(idx_like, arr.shape) if idx_like is not None else (False, False, None)

                if ok and not is_float:
                    # Direct indexing
                    try:
                        if self.mesh.TD == 2:
                            field_data[i_pos, step] = arr[ints[0], ints[1]]
                        else:
                            field_data[i_pos, step] = arr[ints[0], ints[1], ints[2]]
                    except Exception:
                        field_data[i_pos, step] = np.nan
                elif ok and is_float and interpolate:
                    # TODO: Implement bilinear/trilinear interpolation here
                    # Currently fallback to nearest neighbor (already rounded)
                    try:
                        if self.mesh.TD == 2:
                            field_data[i_pos, step] = arr[ints[0], ints[1]]
                        else:
                            field_data[i_pos, step] = arr[ints[0], ints[1], ints[2]]
                    except Exception:
                        field_data[i_pos, step] = np.nan
                else:
                    # If mesh.node_location cannot provide indices in advance, try dynamic request (compatibility fallback)
                    try:
                        idx_now = self.mesh.node_location(pos)
                        ok2, is_float2, ints2 = _idx_to_ints(idx_now, arr.shape)
                        if ok2:
                            if self.mesh.TD == 2:
                                field_data[i_pos, step] = arr[ints2[0], ints2[1]]
                            else:
                                field_data[i_pos, step] = arr[ints2[0], ints2[1], ints2[2]]
                        else:
                            field_data[i_pos, step] = np.nan
                    except Exception:
                        field_data[i_pos, step] = np.nan

        # Plot
        for i, pos in enumerate(positions):
            axes.plot(time_data, field_data[i, :], lw=2, label=f'Position {pos}')

        # xlim
        try:
            axes.set_xlim(time_data[0], time_data[-1])
        except Exception:
            pass

        # ylim: prevent all NaN
        if np.all(np.isnan(field_data)):
            # No data, use default range and warn
            warnings.warn("All position data are NaN, using default y-axis range [-1,1].")
            axes.set_ylim(-1.0, 1.0)
        else:
            mn = float(np.nanmin(field_data))
            mx = float(np.nanmax(field_data))
            if mn == mx:
                # Prevent flat line causing matplotlib errors or compression
                delta = abs(mn) * 0.05 if mn != 0 else 0.5
                axes.set_ylim(mn - delta, mx + delta)
            else:
                axes.set_ylim(mn, mx)

        axes.set_xlabel('Time (s)')
        axes.set_ylabel(f'{field_component} Field')
        axes.legend()
        axes.grid(True)

        return fig, axes

    def _get_field_value_from_array(self, field_array, position: Tuple[float, ...], interpolate: bool = False):
        """
        Get value at specified position from field array - optimized version, 
        compatible with different node_location return types.

        Args:
            position: Physical coordinates or node indices, length should match self.mesh.TD
            interpolate: If True, attempt interpolation when position has float indices
                       (currently implemented as nearest neighbor fallback)
        """
        if field_array is None:
            return np.nan

        # Ensure numpy array
        if hasattr(field_array, 'numpy'):
            arr = field_array.numpy()
        else:
            arr = bm.to_numpy(field_array)

        try:
            idx_like = self.mesh.node_location(position)  # May return tuple of ints or floats
        except Exception as e:
            warnings.warn(f"mesh.node_location({position}) call failed: {e}")
            return np.nan

        # Process idx_like
        try:
            if not isinstance(idx_like, (list, tuple)):
                return np.nan
            # If all are integer types
            if all(isinstance(x, (int, np.integer)) for x in idx_like):
                ints = tuple(int(x) for x in idx_like)
                # Check range
                if self.mesh.TD == 2:
                    i, j = ints
                    if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]:
                        return arr[i, j]
                else:
                    i, j, k = ints
                    if (0 <= i < arr.shape[0] and 0 <= j < arr.shape[1] and 0 <= k < arr.shape[2]):
                        return arr[i, j, k]
                return np.nan
            else:
                # Contains float components, fallback to nearest neighbor (or interpolation implementation point)
                rounded = tuple(int(round(x)) for x in idx_like)
                if self.mesh.TD == 2:
                    i, j = rounded
                    if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]:
                        return arr[i, j]
                else:
                    i, j, k = rounded
                    if (0 <= i < arr.shape[0] and 0 <= j < arr.shape[1] and 0 <= k < arr.shape[2]):
                        return arr[i, j, k]
                return np.nan
        except Exception:
            return np.nan
        
    def show_animation(self,
                   field_history: list,
                   field_component: str = 'Ez',
                   fig: Optional[Figure] = None,
                   axes: Optional[Union[Axes, Axes3D]] = None,
                   fname: str = 'wave_animation.mp4',
                   slice_plane: Optional[tuple] = None,
                   frames: Optional[int] = None,
                   interval: int = 50,
                   plot_type: str = 'imshow',
                   cmap: str = 'viridis',
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None,
                   step: int = 0,
                   verbose: bool = False) -> 'animation.FuncAnimation':
        """
        Optimized animation display function, fixing UPML and object display issues.
        """
        # Frame settings
        if frames is None:
            frames = len(field_history)
        else:
            frames = min(frames, len(field_history))
        
        if frames <= 0:
            raise ValueError("frames<=0 or empty field_history")

        # Create figure and axes
        if fig is None or axes is None:
            if plot_type == 'surface' and getattr(self, 'mesh', None) and getattr(self.mesh, 'TD', 0) == 3:
                fig = plt.figure(figsize=(10, 8))
                axes = fig.add_subplot(111, projection='3d')
            else:
                fig, axes = plt.figure(figsize=(10, 8)), plt.subplot(111)

        # Reset axes
        self._reset_axes(axes)
        for ax in fig.axes[1:]:
            fig.delaxes(ax)

        # Get first frame data for initialization
        first_snapshot = field_history[0]
        first_data = self._get_frame_data_from_snapshot(first_snapshot, field_component, slice_plane)
        
        # Apply slice
        if slice_plane is not None and first_data.ndim == 3:
            first_data = self._apply_slice(first_data, slice_plane)

        # Get data range
        data_min = vmin if vmin is not None else float(np.nanmin(first_data))
        data_max = vmax if vmax is not None else float(np.nanmax(first_data))

        # Get domain information
        domain = getattr(self.pde, 'domain', [0, first_data.shape[1]-1, 0, first_data.shape[0]-1])
        if first_data.ndim == 2:
            extent = [domain[0], domain[1], domain[2], domain[3]]
        else:
            extent = [0, first_data.shape[1]-1, 0, first_data.shape[0]-1]

        # Initialize plot objects
        if first_data.ndim == 1:
            # 1D case
            x = np.arange(frames)
            y = np.full(frames, np.nan, dtype=float)
            line, = axes.plot(x, y, lw=2)
            axes.set_xlim(0, frames-1)
            axes.set_ylim(data_min, data_max)
            axes.set_xlabel('Time Step')
            axes.set_ylabel(field_component)
            artists = [line]
            
        elif plot_type == 'surface' and hasattr(axes, 'plot_surface') and first_data.ndim == 2:
            # 3D surface plot
            X, Y = self._create_mesh_grid_for_slice(first_data.shape, slice_plane)
            surf = axes.plot_surface(X, Y, first_data, cmap=cmap, 
                                vmin=data_min, vmax=data_max, 
                                rstride=1, cstride=1)
            axes.set_xlim(extent[0], extent[1])
            axes.set_ylim(extent[2], extent[3])
            axes.set_zlim(data_min, data_max)
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            axes.set_zlabel(field_component)
            fig.colorbar(surf, ax=axes, shrink=0.6)
            artists = [surf]
            
        else:
            # 2D image display
            img = axes.imshow(first_data, cmap=cmap, vmin=data_min, vmax=data_max,
                            extent=extent, origin='lower', interpolation='none')
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            fig.colorbar(img, ax=axes).set_label(field_component)
            artists = [img]

            # Mark objects
            self._mark_objects(axes, extent)
            
            # Mark UPML boundaries
            self._mark_upml_boundaries(axes, extent)

        # Update function
        def update(n):
            if verbose:
                print(f"Frame {n}/{frames}")
                
            snapshot = field_history[n]
            frame_data = self._get_frame_data_from_snapshot(snapshot, field_component, slice_plane)
            
            if slice_plane is not None and frame_data.ndim == 3:
                frame_data = self._apply_slice(frame_data, slice_plane)

            if first_data.ndim == 1:
                # 1D update
                y[n] = frame_data.ravel()[0] if frame_data.size > 0 else np.nan
                artists[0].set_ydata(y)
                
            elif plot_type == 'surface' and hasattr(axes, 'plot_surface') and frame_data.ndim == 2:
                # 3D surface update
                axes.collections.clear()  # Clear old surface
                new_surf = axes.plot_surface(X, Y, frame_data, cmap=cmap,
                                        vmin=data_min, vmax=data_max,
                                        rstride=1, cstride=1)
                artists[0] = new_surf
                
            else:
                # 2D image update
                artists[0].set_array(frame_data)

            # Update title
            time_val = snapshot.get('time', n * getattr(getattr(self, 'fdtd', None), 'dt', 1.0))
            axes.set_title(f'{field_component} at t={time_val:.2e}s (step {n})')

            # Save single frame
            if step > 0 and n % step == 0:
                base_name = fname.rsplit('.', 1)[0]
                fig.savefig(f"{base_name}_{n:05d}.png")

            return artists

        # Create animation
        ani = animation.FuncAnimation(
            fig, update, frames=frames, interval=interval, 
            blit=(first_data.ndim == 1)  # Only 1D uses blit
        )

        # Save animation
        try:
            ani.save(fname)
        except Exception as e:
            print(f"Failed to save animation: {e}")
            # Fallback to GIF
            gif_name = fname.rsplit('.', 1)[0] + '.gif'
            ani.save(gif_name, writer='pillow')
            print(f"Saved as GIF: {gif_name}")

        return ani

    def _reset_axes(self, axes: Union[Axes, Axes3D], center=True, margin=0.05):
        """Reset axes settings."""
        axes.clear()
        axes.set_aspect('auto')
        axes.autoscale()
        if center:
            axes.margins(margin)

    def _mark_objects(self, axes, extent):
        """Mark object positions on plot, ensuring consistency with material matrices."""
        objects = self.pde.get_object_config()
        if not objects:
            return
        
        for i, obj in enumerate(objects):
            box = obj['box']
            
            if len(box) == 4:  # 2D object
                # Ensure correct coordinate order: [x_min, x_max, y_min, y_max]
                x_min, x_max, y_min, y_max = box
                
                # Validate coordinate range
                if not (extent[0] <= x_min <= x_max <= extent[1] and 
                        extent[2] <= y_min <= y_max <= extent[3]):
                    print(f"Warning: Object {i} coordinates out of range")
                    continue
                
                # Draw rectangle border
                rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                            linewidth=2, edgecolor='red', facecolor='none',
                            alpha=0.7, label=obj.get('tag', 'object'))
                axes.add_patch(rect)

    def _mark_upml_boundaries(self, axes, extent):
        """Mark UPML boundaries on plot."""
        if getattr(getattr(self, 'fdtd', None), 'boundary', '').upper() != 'UPML':
            return
            
        try:
            pml_width = getattr(getattr(self, 'fdtd', None), 'pml_width', 8)
            h = getattr(getattr(self, 'fdtd', None), 'h', 1.0)
            upml_physical = pml_width * h
            
            x_min, x_max, y_min, y_max = extent
            
            # Draw PML regions for four boundaries
            pml_patches = [
                Rectangle((x_min, y_min), upml_physical, y_max-y_min),  # Left
                Rectangle((x_max-upml_physical, y_min), upml_physical, y_max-y_min),  # Right
                Rectangle((x_min, y_min), x_max-x_min, upml_physical),  # Bottom
                Rectangle((x_min, y_max-upml_physical), x_max-x_min, upml_physical)   # Top
            ]
            
            for patch in pml_patches:
                patch.set_linewidth(0)
                patch.set_edgecolor('none')
                patch.set_facecolor((0, 0, 1, 0.1))  # Semi-transparent blue
                axes.add_patch(patch)
                
        except Exception as e:
            print(f"Failed to mark UPML boundaries: {e}")

    def _get_frame_data_from_snapshot(self, snapshot, field_component: str, slice_plane: Optional[Tuple]):
        """Return a numpy array for the requested field component and slice."""
        key = field_component[-1].lower()
        if field_component.startswith('E'):
            field_data = snapshot['E'].get(key)
        else:
            field_data = snapshot['H'].get(key)

        if field_data is None:
            raise ValueError(f"Field component {field_component} not found in snapshot")

        if hasattr(field_data, 'numpy'):
            data = field_data.numpy()
        else:
            data = bm.to_numpy(field_data)

        return self._apply_slice(data, slice_plane)

    def _apply_slice(self, data, slice_plane):
        """Apply slice_plane to a numpy array `data`.

        slice_plane: (axis, position) where axis in {'x','y','z'} and position is int index or float physical coord.
        domain assumed as [x0,x1,y0,y1,z0,z1] when converting physical coord -> index.
        """
        if slice_plane is None:
            return data

        axis, position = slice_plane
        axis = axis.lower()

        if axis not in ('x', 'y', 'z'):
            raise ValueError(f"Invalid slice axis: {axis}")

        if not isinstance(position, int):
            # position is physical coordinate -> convert to index
            domain = getattr(self.pde, 'domain', None)
            if domain is None or len(domain) < 4:
                raise ValueError("Cannot convert physical coordinate to index: domain information missing")

            if axis == 'x':
                n = data.shape[0]
                pos_norm = (position - domain[0]) / (domain[1] - domain[0])
                idx = int(round(pos_norm * (n - 1)))
            elif axis == 'y':
                n = data.shape[1]
                pos_norm = (position - domain[2]) / (domain[3] - domain[2])
                idx = int(round(pos_norm * (n - 1)))
            else:  # z
                if data.ndim < 3:
                    raise ValueError("Data has no z-dimension to slice")
                n = data.shape[2]
                pos_norm = (position - domain[4]) / (domain[5] - domain[4])
                idx = int(round(pos_norm * (n - 1)))
        else:
            idx = int(position)

        # Clamp
        if axis == 'x':
            idx = max(0, min(idx, data.shape[0] - 1))
            return data[idx, :, :]
        elif axis == 'y':
            idx = max(0, min(idx, data.shape[1] - 1))
            return data[:, idx, :]
        else:
            idx = max(0, min(idx, data.shape[2] - 1))
            return data[:, :, idx]

    def _create_mesh_grid_for_slice(self, data_shape, slice_plane: Optional[Tuple]):
        """Create X,Y mesh arrays matching a `data_shape` (rows,cols) for plotting.

        Attempts to use mesh.node_coords if available; falls back to linspace over domain.
        Returns X,Y suitable for axes.plot_surface or pcolormesh (indexing='xy').
        """
        domain = getattr(self.pde, 'domain', None)

        # Try mesh node_coords if it matches
        try:
            if hasattr(self.mesh, 'node_coords'):
                node_coords = getattr(self.mesh, 'node_coords')
                if node_coords is not None:
                    # node_coords might be (ny, nx, 2) or (nx, ny, 2)
                    if node_coords.ndim == 3 and node_coords.shape[2] >= 2:
                        # Try to find orientation that matches data_shape
                        for attempt in [(0, 1), (1, 0)]:
                            A = node_coords.shape[attempt[0]]
                            B = node_coords.shape[attempt[1]]
                            if (A, B) == tuple(data_shape):
                                if attempt == (0, 1):
                                    X = node_coords[:, :, 0]
                                    Y = node_coords[:, :, 1]
                                else:
                                    X = node_coords[:, :, 0].T
                                    Y = node_coords[:, :, 1].T
                                return X, Y
        except Exception:
            pass

        # Fallback: use domain linspace or indices
        if domain is not None and len(domain) >= 4 and data_shape[0] > 1 and data_shape[1] > 1:
            x = np.linspace(domain[0], domain[1], data_shape[1])
            y = np.linspace(domain[2], domain[3], data_shape[0])
            X, Y = np.meshgrid(x, y, indexing='xy')
            return X, Y

        # Last fallback: integer grid
        x = np.arange(data_shape[1])
        y = np.arange(data_shape[0])
        X, Y = np.meshgrid(x, y, indexing='xy')
        return X, Y

    def _mark_upml_boundaries(self, axes, box: List[float]):
        """Mark UPML boundary rectangles on axes.

        `box` must be in the form [xmin, xmax, ymin, ymax, vmin, vmax]
        """
        try:
            if getattr(self.fdtd, 'boundary', '').upper() == 'UPML':
                xmin, xmax, ymin, ymax = box[0], box[1], box[2], box[3]
                x_len = xmax - xmin
                y_len = ymax - ymin
                upml_width = int(getattr(self.fdtd, 'pml_width', 8))
                # Try to get physical grid spacing h from fdtd or mesh
                h = getattr(self.fdtd, 'h', None)
                if h is None:
                    # Mesh may expose grid spacing
                    try:
                        hx = (xmax - xmin) / max(1, getattr(self.mesh, 'nx', 1))
                        hy = (ymax - ymin) / max(1, getattr(self.mesh, 'ny', 1))
                        h = min(hx, hy)
                    except Exception:
                        h = 1.0
                upml = upml_width * float(h)

                rects = [
                    (xmin, ymin, upml, y_len),
                    (xmax - upml, ymin, upml, y_len),
                    (xmin, ymin, x_len, upml),
                    (xmin, ymax - upml, x_len, upml)
                ]

                for x0, y0, w, hrect in rects:
                    axes.add_patch(Rectangle((x0, y0), w, hrect, linewidth=0, edgecolor='none', facecolor=(0, 0.7, 1.0, 0.12)))
        except Exception:
            # Don't fail visualization for UPML marking
            pass

    def __str__(self):
        """Model information summary."""
        s = f"{self.__class__.__name__}(\n"
        s += f"  PDE            : {self.pde.__class__.__name__}\n"
        s += f"  Dimension      : {self.mesh.TD}D\n"
        s += f"  Domain         : {self.pde.domain}\n"
        s += f"  Grid size      : {self.mesh.nx} x {self.mesh.ny}"
        if self.mesh.TD == 3:
            s += f" x {self.mesh.nz}"
        s += f"\n  Time step      : {self.fdtd.dt:.2e}s\n"
        s += f"  Courant number : {self.fdtd.R:.4f}\n"
        s += f"  Boundary       : {self.fdtd.boundary}\n"
        s += f"  Sources        : {len(self.source_manager.sources)}\n"
        s += f"  Current time   : {self.current_time:.2e}s (step {self.current_step})\n"
        s += ")"
        return s