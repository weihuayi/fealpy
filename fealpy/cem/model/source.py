from typing import Tuple, Optional, Callable, Any, Union, List, Dict
from fealpy.backend import backend_manager as bm
import math

# Physical constants
c0 = 299792458.0
mu0 = 4.0 * math.pi * 1e-7
eta0: float = mu0 * c0

# ---------------------------
# Common time-domain waveforms (callable: t -> amplitude)
# ---------------------------

def gaussian_pulse(t: float, t0: float = 1.0, tau: float = 0.2) -> float:
    """Gaussian pulse: exp(-((t-t0)/tau)**2)."""
    return math.exp(-((t - t0) / tau) ** 2)

def ricker_wavelet(t: float, t0: float = 1.0, f: float = 1.0) -> float:
    """Ricker wavelet (Mexican hat), commonly used as point pulse source.
    
    Args:
        t: Time
        t0: Time offset
        f: Center frequency
        
    Returns:
        Waveform amplitude at time t
    """
    x = math.pi * f * (t - t0)
    x2 = x * x
    return (1.0 - 2.0 * x2) * math.exp(-x2)

def sinusoid(t: float, freq: float = 1.0, phase: float = 0.0) -> float:
    """Sinusoidal waveform."""
    return math.sin(2.0 * math.pi * freq * t + phase)

def gaussian_enveloped_sine(t: float, freq: float = 1.0, t0: float = 1.0, tau: float = 0.2) -> float:
    """Gaussian enveloped sine wave."""
    return math.exp(-((t - t0) / tau) ** 2) * math.sin(2.0 * math.pi * freq * (t - t0))

# ---------------------------
# Source class and manager
# ---------------------------

class Source:
    """
    Time-domain point/small-region injection source (adapted for Yee grid).
    
    Design goals:
      - Support injection to single component of E or H ('Ex','Ey','Ez','Hx','Hy','Hz')
      - Support both 'soft' (accumulative) and 'hard' (forced setting) injection methods
      - Support two spatial positioning modes: grid indices (i,j[,k]) or physical coordinates (x,y[,z])
      - Support small-region expansion (spread_radius, in grid points) to avoid single-point singularity
    
    Note:
      - In physical coordinate mode, the module attempts to map coordinates to the most appropriate 
        grid point (for E components mapped to nodes; for H components approximate mapping).
        For precise H component mapping, it is recommended to directly pass grid indices.
    """

    def __init__(self,
                 position: Union[Tuple[int, ...], Tuple[float, ...]],
                 comp: str = "Ez",
                 waveform: Optional[Callable[[float], float]] = None,
                 injection: str = "soft",
                 amplitude: float = 1.0,
                 spread: int = 0,
                 use_normalized_amplitude: bool = True):
        """
        Initialize source.

        Args:
            position: If int tuple -> treated as grid indices (i,j[,k]);
                      If float tuple -> treated as physical coordinates (x,y[,z])
            comp: 'Ex','Ey','Ez','Hx','Hy','Hz' (case sensitive)
            waveform: Callable t -> scalar. If None, uses unit impulse (returns 1.0)
            injection: 'soft' (accumulate) or 'hard' (overwrite)
            amplitude: Global scaling factor (can be used for volume/area scaling)
            spread: Spatial spread radius (in grid points), 0 means single point injection
            use_normalized_amplitude: Whether to use dimensionless amplitude 
                                     (should be True in dimensionless FDTD systems)
        """
        assert comp in {"Ex","Ey","Ez","Hx","Hy","Hz"}, "comp must be one of Ex/Ey/Ez/Hx/Hy/Hz"
        assert injection in {"soft","hard"}, "injection must be 'soft' or 'hard'"

        self.position = position
        self.comp = comp
        self.waveform = waveform if waveform is not None else (lambda t: 1.0)
        self.injection = injection
        self.amplitude = float(amplitude)
        self.spread = int(spread)
        self.use_normalized_amplitude = use_normalized_amplitude

    # === Spatial mapping helper methods ===
    def _is_index_position(self) -> bool:
        """Check if position is given as integer indices."""
        return all(isinstance(v, int) for v in self.position)

    def _is_coord_position(self) -> bool:
        """Check if position is given as physical coordinates."""
        return all(isinstance(v, float) or isinstance(v, int) for v in self.position)

    def _map_coord_to_index(self, yee_mesh) -> Tuple[int, ...]:
        """
        Map physical coordinates to grid indices (prioritizing E-corresponding node points).
        
        Returns:
            Index tuple (i,j[,k])
            
        Raises:
            ValueError: If position dimension doesn't match mesh dimension
        """
        pos = self.position
        TD = getattr(yee_mesh, "TD", 2)
        
        # Ensure coordinate dimension matches mesh dimension
        if len(pos) != TD:
            raise ValueError(f"Position dimension {len(pos)} doesn't match mesh dimension {TD}")
            
        coords = [float(p) for p in pos]
        
        try:
            idx_tuple = yee_mesh.node_location(bm.asarray([coords], dtype=yee_mesh.mesh.ftype, device=yee_mesh.device))
            
            # More robust handling: ensure returned index count matches dimension
            if len(idx_tuple) != TD:
                raise ValueError(f"node_location returned {len(idx_tuple)} indices for {TD}D mesh")
                
            idxs = []
            for i, a in enumerate(idx_tuple):
                a_array = bm.asarray(a)
                if hasattr(a_array, "item"):
                    idx_val = int(a_array.item())
                else:
                    idx_val = int(a_array[0])
                idxs.append(idx_val)
                
            return tuple(idxs)
            
        except Exception as e:
            # Fallback: rough calculation based on grid spacing
            h = yee_mesh.h
            origin = yee_mesh.origin
            
            if TD == 2:
                i = int(round((coords[0] - origin[0]) / h[0]))
                j = int(round((coords[1] - origin[1]) / h[1]))
                return (i, j)
            else:  # 3D
                i = int(round((coords[0] - origin[0]) / h[0]))
                j = int(round((coords[1] - origin[1]) / h[1]))
                k = int(round((coords[2] - origin[2]) / h[2]))
                return (i, j, k)

    def _neighbour_index_list(self, center_idx: Tuple[int, ...], yee_mesh) -> List[Tuple[int, ...]]:
        """Return index list centered at center_idx with spread radius (considering boundary clipping)."""
        if self.spread <= 0:
            return [center_idx]
            
        TD = getattr(yee_mesh, "TD", 2)
        nx, ny = yee_mesh.nx, yee_mesh.ny
        nz = getattr(yee_mesh, "nz", 0)
        out = []
        
        # Build ranges based on dimension
        ranges = []
        dim_limits = [nx, ny, nz] if TD == 3 else [nx, ny]
        
        for d, c in enumerate(center_idx):
            maxd = dim_limits[d]
            lo = max(0, c - self.spread)
            hi = min(maxd, c + self.spread)
            ranges.append(range(lo, hi + 1))
            
        # Generate index list
        if TD == 2:
            for ii in ranges[0]:
                for jj in ranges[1]:
                    out.append((ii, jj))
        else:  # 3D
            for ii in ranges[0]:
                for jj in ranges[1]:
                    for kk in ranges[2]:
                        out.append((ii, jj, kk))
        return out

    # === Weight function ===
    def _distance_weight(self, center_coord, coord, TD: int) -> float:
        """Calculate distance weight, properly handling 2D and 3D cases."""
        dx = coord[0] - center_coord[0]
        dy = coord[1] - center_coord[1] 
        dz = coord[2] - center_coord[2] if TD == 3 else 0.0
        
        r = math.sqrt(dx*dx + dy*dy + dz*dz)
        return r

    # === Main injection function (external call) ===
    def apply(self, t: float, yee_mesh, E_fields: Dict[str, Any], H_fields: Dict[str, Any]) -> None:
        """
        Apply current source to E_fields / H_fields at time t (modifies arrays in-place).
        
        Args:
            t: Current time
            yee_mesh: Yee grid mesh object
            E_fields: Dictionary of E field components
            H_fields: Dictionary of H field components
        """
        TD = getattr(yee_mesh, "TD", 2)
        
        # Calculate base amplitude
        base_amp = float(self.waveform(t)) * self.amplitude

        # If using dimensionless system and electric field source, normalize amplitude
        if self.use_normalized_amplitude and self.comp.startswith("E"):
            amp = base_amp / eta0
        else:
            amp = base_amp
            
        if amp == 0.0:
            return

        # Determine index center
        if self._is_index_position():
            center_idx = tuple(int(v) for v in self.position)
        elif self._is_coord_position():
            center_idx = self._map_coord_to_index(yee_mesh)
        else:
            raise ValueError("Unsupported position specification")

        # Construct index list for writing
        idxs = self._neighbour_index_list(center_idx, yee_mesh)

        # If spread>0, calculate weights for each index
        weights = []
        if self.spread > 0:
            # Get physical center coordinates
            try:
                node_shape = [yee_mesh.nx + 1, yee_mesh.ny + 1]
                if TD == 3:
                    node_shape.append(getattr(yee_mesh, "nz", 0) + 1)
                node_shape.append(TD)
                
                node_coords = yee_mesh.mesh.node.reshape(*node_shape).squeeze()
                center_coord = tuple(float(node_coords[tuple(center_idx)][d]) for d in range(TD))
            except Exception:
                # Fallback: calculate from origin + index * h
                origin = yee_mesh.origin
                center_coord = tuple(origin[d] + center_idx[d] * yee_mesh.h[d] for d in range(TD))

            total_w = 0.0
            for idx in idxs:
                try:
                    coord = tuple(float(node_coords[tuple(idx)][d]) for d in range(TD))
                except Exception:
                    coord = tuple(origin[d] + idx[d] * yee_mesh.h[d] for d in range(TD))
                    
                r = self._distance_weight(center_coord, coord, TD)
                # Simple inverse distance weight, truncated at spread * h
                R = max(1e-12, self.spread * sum(yee_mesh.h) / TD)
                if r > R:
                    w = 0.0
                else:
                    s = r / (R + 1e-15)
                    w = (1.0 - s**3)**3
                weights.append(w)
                total_w += w
                
            if total_w <= 0:
                weights = [1.0 / len(idxs)] * len(idxs)
            else:
                weights = [w / total_w for w in weights]
        else:
            weights = [1.0] * len(idxs)

        # Select target array (E or H)
        target_is_E = self.comp.startswith("E")
        comp_char = self.comp[1].lower()  # 'x','y','z'
        if target_is_E:
            arr = E_fields.get(comp_char)
        else:
            arr = H_fields.get(comp_char)

        if arr is None:
            return

        # Execute injection
        for (idx, w) in zip(idxs, weights):
            injection_value = amp * float(w)
            try:
                if arr.ndim == 2:  # 2D
                    i, j = idx
                    if not (0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]):
                        continue
                    if self.injection == "soft":
                        arr[i, j] += injection_value
                    else:
                        arr[i, j] = injection_value
                        
                elif arr.ndim == 3:  # 3D
                    i, j, k = idx
                    if not (0 <= i < arr.shape[0] and 0 <= j < arr.shape[1] and 0 <= k < arr.shape[2]):
                        continue
                    if self.injection == "soft":
                        arr[i, j, k] += injection_value
                    else:
                        arr[i, j, k] = injection_value
                else:
                    continue
            except Exception as e:
                # Alternative: use safer assignment method
                print(f"Warning: Source injection failed at {idx}: {e}")
                continue


class SourceManager:
    """
    Container for managing multiple Source objects.
    """

    def __init__(self, use_normalized_system: bool = True):
        """
        Initialize source manager.
        
        Args:
            use_normalized_system: Whether to use normalized system
        """
        self.sources: List[Source] = []
        self.use_normalized_system = use_normalized_system

    def add(self, src: Source) -> None:
        """
        Add a source to the manager.
        
        Args:
            src: Source to add
        """
        # Ensure newly added source uses correct normalization setting
        if self.use_normalized_system:
            src.use_normalized_amplitude = True
        self.sources.append(src)

    def remove(self, src: Source) -> None:
        """
        Remove a source from the manager.
        
        Args:
            src: Source to remove
        """
        self.sources.remove(src)

    def clear(self) -> None:
        """Clear all sources."""
        self.sources.clear()

    def apply_all(self, t: float, yee_mesh, E_fields: Dict[str, Any], H_fields: Dict[str, Any]) -> None:
        """
        Apply all sources at time t.
        
        Args:
            t: Current time
            yee_mesh: Yee grid mesh object
            E_fields: Dictionary of E field components
            H_fields: Dictionary of H field components
        """
        for src in self.sources:
            src.apply(t, yee_mesh, E_fields, H_fields)