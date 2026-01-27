from typing import Optional, Tuple, List, Dict, Any, Sequence
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian

# Physical constants
c0 = 299792458.0
mu0 = 4.0 * bm.pi * 1e-7
eps0 = 1.0 / (mu0 * c0 * c0)


class PointSourceMaxwell:
    """
    PDE model: point source problem for time-domain Maxwell equations.
    
    This model describes electromagnetic wave propagation in homogeneous or 
    inhomogeneous media, supporting point source excitation and material 
    parameter settings for cuboid object regions.
    
    Attributes:
        domain (Sequence[float]): Solution domain in format [xmin, xmax, ymin, ymax] (2D) 
                                  or [xmin, xmax, ymin, ymax, zmin, zmax] (3D)
        eps (float): Relative permittivity
        mu (float): Relative permeability  
        geo_dimension (int): Geometric dimension
    """

    def __init__(
        self,
        eps: float = 1.0,
        mu: float = 1.0,
        domain: Optional[Sequence[float]] = None,
    ) -> None:
        """
        Initialize point source Maxwell equation model.

        Parameters:
            eps: Relative permittivity, default 1.0 (vacuum)
            mu: Relative permeability, default 1.0 (vacuum)
            domain: Solution domain, default 3D unit cube [0,1,0,1,0,1]
        """
        # Use default 3D unit cube if domain not specified
        self._domain = domain if domain is not None else [0, 1, 0, 1, 0, 1]

        # Calculate absolute permittivity and permeability
        self.c0 = 299792458.0
        self.eps = eps
        self.mu = mu
        self._dim = int(len(self._domain) / 2)
        
        # Source configuration collection (PDE only stores metadata)
        self._sources: List[Dict[str, Any]] = []
        self._objects: List[Dict[str, Any]] = []

        # Tag counter (for default naming)
        self._source_counter = 0
        self._object_counter = 0

    @property
    def domain(self) -> Sequence[float]:
        """Return solution domain."""
        return self._domain

    @property
    def geo_dimension(self) -> int:
        """Return geometric dimension."""
        return self._dim

    def permittivity(self) -> float:
        """Return absolute permittivity ε(x,y)."""
        return eps0 * self.eps

    def permeability(self) -> float:
        """Return absolute permeability μ(x,y)."""
        return mu0 * self.mu
    
    def add_source(
        self,
        position: Optional[Tuple[float, ...]] = None,
        comp: str = "Ez",
        waveform: str = "sinusoid",
        waveform_params: Optional[Dict[str, float]] = None,
        amplitude: float = 1.0,
        spread: int = 0,
        injection: str = "soft",
        tag: Optional[str] = None,
    ) -> str:
        """
        Add a point source to the PDE model (only metadata stored).

        Parameters:
            position: Physical coordinates (x,y) or (x,y,z). If None, uses domain center
            comp: Injection component name, e.g., 'Ez' (2D) or 'Ex'/'Ey'/'Ez' (3D)
            waveform: Waveform name, e.g., 'gaussian', 'ricker', 'sinusoid', 'gaussian_enveloped_sine'
            waveform_params: Waveform parameter dictionary (e.g., freq, t0, tau, phase)
            amplitude: Amplitude (scalar)
            spread: Spatial spread (in grid points/half-width units)
            injection: Injection preference 'soft' (superposition) or 'hard' (overwrite)
            tag: Unique source tag, auto-generated as 'src_<n>' if None

        Returns:
            tag: Unique source tag

        Raises:
            ValueError: When parameters are invalid
        """
        # Default waveform parameters
        wp = dict(waveform_params) if waveform_params is not None else {}
        # Default position: domain center
        if position is None:
            position = self._domain_center()

        if tag is None:
            tag = f"src_{self._source_counter}"
            self._source_counter += 1

        src_cfg: Dict[str, Any] = {
            "tag": tag,
            "position": tuple(float(x) for x in position),
            "comp": str(comp),
            "waveform": str(waveform),
            "waveform_params": wp,
            "amplitude": float(amplitude),
            "spread": int(spread),
            "injection": str(injection),
            "dim": self.geo_dimension,
        }

        # Append to sources list
        self._sources.append(src_cfg)
        return tag

    def set_sources(self, sources: List[Dict[str, Any]]) -> None:
        """
        Set multiple sources at once (replace all existing sources).

        Parameters:
            sources: List of source configuration dictionaries compatible with add_source format

        Raises:
            ValueError: When source configuration lacks required keys
        """
        # Basic structure validation
        for s in sources:
            if "position" not in s or "comp" not in s:
                raise ValueError("each source dict must contain at least 'position' and 'comp' keys")
        self._sources = [dict(s) for s in sources]  # shallow copy
        # Reset counter to avoid naming conflicts
        self._source_counter = len(self._sources)

    def remove_source(self, tag: str) -> bool:
        """
        Remove source by tag.

        Parameters:
            tag: Source tag to remove

        Returns:
            True if removal successful, False otherwise
        """
        for i, s in enumerate(self._sources):
            if s.get("tag") == tag:
                del self._sources[i]
                return True
        return False

    def list_sources(self) -> List[Dict[str, Any]]:
        """Return shallow copy of all current source configurations."""
        return [dict(s) for s in self._sources]

    def get_source_config(self) -> List[Dict[str, Any]]:
        """
        Return source configuration list for computational model.

        Returns:
            List of source configuration dictionaries
        """
        return [dict(s) for s in self._sources]

    def add_object(
        self,
        box: Sequence[float],
        eps: Optional[float] = None,
        mu: Optional[float] = None,
        conductivity: float = 0.0,
        tag: Optional[str] = None,
    ) -> str:
        """
        Add a cuboid object region to the PDE model.
        
        Parameters:
            box: Cuboid region in format [xmin, xmax, ymin, ymax] (2D) 
                 or [xmin, xmax, ymin, ymax, zmin, zmax] (3D)
            eps: Relative permittivity, uses background value if None
            mu: Relative permeability, uses background value if None  
            conductivity: Electrical conductivity (S/m), default 0 (ideal medium)
            tag: Unique object tag, auto-generated as 'obj_<n>' if None
        
        Returns:
            tag: Object tag
            
        Raises:
            ValueError: When box dimension mismatch or invalid range
        """
        # Validate box dimension
        if len(box) != 2 * self.geo_dimension:
            raise ValueError(f"box must have {2 * self.geo_dimension} elements for {self.geo_dimension}D domain")
        
        # Validate box range
        for i in range(0, len(box), 2):
            if box[i] >= box[i + 1]:
                raise ValueError(f"box range invalid: {box[i]} >= {box[i + 1]}")
        
        # Generate default tag
        if tag is None:
            tag = f"obj_{self._object_counter}"
            self._object_counter += 1
        
        obj_cfg: Dict[str, Any] = {
            "tag": tag,
            "type": "box",
            "box": [float(x) for x in box],
            "eps": float(eps) if eps is not None else None,
            "mu": float(mu) if mu is not None else None,
            "conductivity": float(conductivity),
            "dim": self.geo_dimension,
        }
        
        self._objects.append(obj_cfg)
        return tag

    def set_objects(self, objects: List[Dict[str, Any]]) -> None:
        """
        Set multiple objects at once (replace all existing objects).
        
        Parameters:
            objects: List of object configuration dictionaries, each must contain 'box' key
            
        Raises:
            ValueError: When object configuration lacks required keys
        """
        for obj in objects:
            if "box" not in obj:
                raise ValueError("each object dict must contain 'box' key")
        
        self._objects = [dict(obj) for obj in objects]
        self._object_counter = len(self._objects)

    def remove_object(self, tag: str) -> bool:
        """
        Remove object by tag.
        
        Parameters:
            tag: Object tag to remove
            
        Returns:
            True if removal successful, False otherwise
        """
        for i, obj in enumerate(self._objects):
            if obj.get("tag") == tag:
                del self._objects[i]
                return True
        return False

    def list_objects(self) -> List[Dict[str, Any]]:
        """Return shallow copy of all current object configurations."""
        return [dict(obj) for obj in self._objects]

    def get_object_config(self) -> List[Dict[str, Any]]:
        """
        Return object configuration list for computational model.
        
        Returns:
            List of object configuration dictionaries
        """
        return [dict(obj) for obj in self._objects]

    def clear_objects(self) -> None:
        """Clear all object configurations."""
        self._objects.clear()
        self._object_counter = 0

    def _domain_center(self) -> Tuple[float, ...]:
        """Domain center coordinates (returns 2 or 3-tuple based on geo_dimension)."""
        if self.geo_dimension == 2:
            x0, x1, y0, y1 = self._domain
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        else:
            x0, x1, y0, y1, z0, z1 = self._domain
            return ((x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0)

    def __str__(self) -> str:
        """
        Return formatted string with model and point source configuration.
        
        Returns:
            Formatted model information string
        """
        s = []
        s.append("PointSourceMaxwell PDE model (PDE layer: metadata only)")
        s.append(f"  Geometry dimension : {self.geo_dimension}D")
        s.append(f"  Domain (flat list) : {self._domain}")
        s.append(f"  Background eps, mu : eps_r={self.eps}, mu_r={self.mu}")
        s.append(f"  Absolute eps, mu   : eps={self.permittivity():.6e}, mu={self.permeability():.6e}")
        
        s.append("  --- sources (metadata) ---")
        if not self._sources:
            s.append("  (no sources configured)")
        else:
            for src in self._sources:
                s.append(
                    f"  tag: {src.get('tag')}, pos: {src.get('position')}, comp: {src.get('comp')}, "
                    f"waveform: {src.get('waveform')}, amplitude: {src.get('amplitude')}, spread: {src.get('spread')}"
                )
        
        s.append("  --- objects ---")
        if not self._objects:
            s.append("  (no objects configured)")
        else:
            for obj in self._objects:
                obj_info = f"  tag: {obj.get('tag')}, box: {obj.get('box')}"
                if obj.get('eps') is not None:
                    obj_info += f", eps_r: {obj.get('eps')}"
                if obj.get('mu') is not None:
                    obj_info += f", mu_r: {obj.get('mu')}"
                if obj.get('conductivity', 0) != 0:
                    obj_info += f", conductivity: {obj.get('conductivity')}"
                s.append(obj_info)
                
        return "\n".join(s)

    # Maintain backward compatibility
    def summary(self) -> str:
        """Return model summary (compatible with legacy interface)."""
        return self.__str__()