from typing import Tuple, Optional, Callable, Any, Union, List, Dict
from fealpy.backend import backend_manager as bm

class YeeUniformMesher:
    """
    Yee grid utility class based on UniformMesh.
    
    Constructed by passing a UniformMesh instance (or any mesh object with compatible attributes).
    
    Attributes:
        mesh: Underlying uniform mesh
        TD: Topological dimension (2 or 3)
        nx, ny, nz: Grid dimensions
        h: Grid spacing tuple
        origin: Origin coordinates
        ftype: Floating point type
        itype: Integer type
        device: Computation device
        node: Node coordinates
        _coordinate_cache: Cache for coordinate calculations
        pml: PML configuration dictionary
    """

    def __init__(self,
                 domain: Tuple[float, ...] = (0.0, 1.0, 0.0, 1.0),
                 nx: int = 50,
                 ny: int = 50,
                 nz: int = 0):
        """
        Initialize Yee uniform mesher.

        Args:
            domain: Domain boundaries (xmin,xmax,ymin,ymax) for 2D or 
                   (xmin,xmax,ymin,ymax,zmin,zmax) for 3D
            nx: Number of grid points in x-direction
            ny: Number of grid points in y-direction  
            nz: Number of grid points in z-direction (0 for 2D)
        """
        from fealpy.mesh import UniformMesh

        if int(nz):
            # Domain must be (xmin,xmax,ymin,ymax,zmin,zmax)
            mesh = UniformMesh(domain, [0, nx, 0, ny, 0, nz])
        else:
            # Domain must be (xmin,xmax,ymin,ymax)
            mesh = UniformMesh(domain, [0, nx, 0, ny])
        
        # Convenience shortcuts
        self.mesh = mesh
        self.TD = getattr(mesh, 'TD', None)
        assert self.TD in (2, 3), "Only 2D or 3D meshes are supported"
        self.nx = getattr(mesh, 'nx')
        self.ny = getattr(mesh, 'ny')
        self.nz = getattr(mesh, 'nz', 0)
        self.h = tuple(getattr(mesh, 'h'))
        self.origin = tuple(getattr(mesh, 'origin'))
        self.ftype = getattr(mesh, 'ftype', None)
        self.itype = getattr(mesh, 'itype', None)
        self.device = getattr(mesh, 'device', None)
        
        # If UniformMesh.node is bm.ndarray then use directly, otherwise try to convert
        self.node = getattr(mesh, 'node')

        # Store field data containers (can be replaced when interacting with TimeDomain nodes)
        self._coordinate_cache = {}
        
        # Additional information
        self.pml = {}

    @property
    def node_coords(self) -> Any:
        """Node coordinates."""
        return self.node.reshape(self.nx + 1, self.ny + 1, getattr(self, "nz", 0) + 1, self.TD).squeeze()

    @property
    def edgex_coords(self) -> Any:
        """x-edge center coordinates."""
        if 'edgex' not in self._coordinate_cache:
            self._coordinate_cache['edgex'] = self.mesh.edgex_barycenter()
        return self._coordinate_cache['edgex']

    @property
    def edgey_coords(self) -> Any:
        """y-edge center coordinates."""
        if 'edgey' not in self._coordinate_cache:
            self._coordinate_cache['edgey'] = self.mesh.edgey_barycenter()
        return self._coordinate_cache['edgey']

    @property
    def edgez_coords(self) -> Any:
        """z-edge center coordinates (3D only)."""
        if self.TD != 3:
            raise ValueError("edgez_coords only available in 3D")
        if 'edgez' not in self._coordinate_cache:
            self._coordinate_cache['edgez'] = self.mesh.edgez_barycenter()
        return self._coordinate_cache['edgez']

    @property
    def facex_coords(self) -> Any:
        """x-face center coordinates (3D only)."""
        if self.TD != 3:
            raise ValueError("facex_coords only available in 3D")
        if 'facex' not in self._coordinate_cache:
            self._coordinate_cache['facex'] = self.mesh.facex_barycenter()
        return self._coordinate_cache['facex']

    @property
    def facey_coords(self) -> Any:
        """y-face center coordinates (3D only)."""
        if self.TD != 3:
            raise ValueError("facey_coords only available in 3D")
        if 'facey' not in self._coordinate_cache:
            self._coordinate_cache['facey'] = self.mesh.facey_barycenter()
        return self._coordinate_cache['facey']

    @property
    def facez_coords(self) -> Any:
        """z-face center coordinates (3D only)."""
        if self.TD != 3:
            raise ValueError("facez_coords only available in 3D")
        if 'facez' not in self._coordinate_cache:
            self._coordinate_cache['facez'] = self.mesh.facez_barycenter()
        return self._coordinate_cache['facez']

    @property
    def cell_coords(self) -> Any:
        """Cell center coordinates."""
        if 'cell' not in self._coordinate_cache:
            self._coordinate_cache['cell'] = self.mesh.cell_barycenter()
        return self._coordinate_cache['cell']
    
    def cell_location(self, p: Any) -> Tuple:
        """
        Return cell indices (i,j[,k]) containing points p.

        Args:
            p: Points to locate in the mesh
            
        Returns:
            Tuple of integer index arrays (one per dimension)
            
        Raises:
            ValueError: If mesh dimension is not supported
        """
        TD = self.TD
        if TD not in (2, 3):
            raise ValueError(f"Unsupported mesh dimension: TD = {TD}")

        ftype = self.ftype
        itype = self.itype
        device = self.device

        h = bm.array(self.h, dtype=ftype, device=device)
        origin = self.origin

        p = bm.asarray(p, dtype=ftype, device=device)
        v = p - origin

        epsilon = 1e-10
        floored = bm.floor(v / h + epsilon)
        indices = bm.astype(floored, itype, device=device)

        return tuple(indices[..., d] for d in range(TD))

    def node_location(self, p: Any) -> Tuple:
        """
        Return nearest node indices for points p.

        Args:
            p: Points to locate in the mesh
            
        Returns:
            Tuple of integer index arrays (one per dimension)
            
        Raises:
            ValueError: If mesh dimension is not supported
        """
        TD = self.TD
        if TD not in (2, 3):
            raise ValueError(f"Only 2D or 3D meshes are supported. Got TD = {TD}")

        ftype = self.ftype
        itype = self.itype
        device = self.device

        h = bm.asarray(self.h, dtype=ftype, device=device)
        origin = self.origin

        p = bm.asarray(p, dtype=ftype, device=device)
        v = p - origin

        rounded = bm.where(v >= 0, bm.floor(v / h + 0.5), bm.ceil(v / h - 0.5))
        indices = bm.astype(rounded, itype, device=device)
        return tuple(indices[..., d] for d in range(TD))

    def interpolation(self, f: Callable, intertype: str = "node") -> Any:
        """
        Evaluate function f at mesh-centered locations.

        Args:
            f: Function to evaluate
            intertype: Type of interpolation location ('node', 'cell', 'edgex', 'edgey', 
                      'edgez', 'facex', 'facey', 'facez', 'face', 'edge')
                      
        Returns:
            Interpolated values at specified locations
            
        Raises:
            ValueError: If interpolation type is unknown
        """
        mesh = self.mesh

        if intertype == "cell":
            return f(self.cell_coords())
        if intertype == "node":
            node = self.node.reshape(self.nx + 1, self.ny + 1, getattr(mesh, "nz", 0) + 1, self.TD).squeeze()
            return f(node)
        if intertype == "facex":
            return f(self.facex_coords)
        if intertype == "facey":
            return f(self.facey_coords)
        if intertype == "facez":
            return f(self.facez_coords)
        if intertype == "face":
            xbc, ybc, zbc = mesh.face_barycenter()
            return f(xbc), f(ybc), f(zbc)
        if intertype == "edgex":
            return f(self.edgex_coords)
        if intertype == "edgey":
            return f(self.edgey_coords)
        if intertype == "edgez":
            return f(self.edgez_coords)
        if intertype == "edge":
            bary = mesh.edge_barycenter()
            return tuple(f(b) for b in bary)

        raise ValueError(f"Unknown interpolation type: {intertype}")
    
    def get_field_matrix(self, etype: str = "node", axis: Optional[str] = None) -> Any:
        """
        Get field matrix of appropriate shape for given entity type and axis.

        Args:
            etype: Entity type ('node', 'cell', 'edge', 'face')
            axis: Specific axis ('x', 'y', 'z') or None
            
        Returns:
            Zero-initialized field matrix of appropriate shape
            
        Raises:
            ValueError: If parameters are invalid for the mesh dimension
        """
        mesh = self.mesh
        TD = self.TD
        dtype = getattr(mesh, "ftype", None)
        nx, ny = self.nx, self.ny
        nz = getattr(self, "nz", 0)

        Z = lambda shape: bm.zeros(shape, dtype=dtype, device=self.device)

        if TD == 2:
            if etype == "edge" and axis is None:
                return Z((nx + 1, ny)), Z((nx, ny + 1))
            shapes_2d = {
                ("node", None): (nx + 1, ny + 1),
                ("node", "z"): (nx + 1, ny + 1),
                ("cell", None): (nx, ny),
                ("edge", "x"): (nx + 1, ny),
                ("edge", "y"): (nx, ny + 1),
            }
            key = (etype, axis)
            if key not in shapes_2d:
                raise ValueError(f"Invalid etype='{etype}' or axis='{axis}' for TD=2")
            return Z(shapes_2d[key])

        if TD != 3:
            raise ValueError(f"Unsupported mesh.TD={TD}")

        if etype == "face" and axis is None:
            return Z((nx + 1, ny, nz)), Z((nx, ny + 1, nz)), Z((nx, ny, nz + 1))
        if etype == "edge" and axis is None:
            return Z((nx, ny + 1, nz + 1)), Z((nx + 1, ny, nz + 1)), Z((nx + 1, ny + 1, nz))

        shapes_3d = {
            ("node", None): (nx + 1, ny + 1, nz + 1),
            ("cell", None): (nx, ny, nz),
            ("face", "x"): (nx + 1, ny, nz),
            ("face", "y"): (nx, ny + 1, nz),
            ("face", "z"): (nx, ny, nz + 1),
            ("edge", "x"): (nx, ny + 1, nz + 1),
            ("edge", "y"): (nx + 1, ny, nz + 1),
            ("edge", "z"): (nx + 1, ny + 1, nz),
        }
        key = (etype, axis)
        if key not in shapes_3d:
            raise ValueError(f"Invalid etype='{etype}' or axis='{axis}' for TD=3")
        return Z(shapes_3d[key])
    
    def initialize_field(self, field_name: str, func: Callable, dt: float, 
                    times: Optional[Union[int, List[float]]] = None) -> Union[Any, Tuple[Any, Any]]:
        """
        Initialize a single field component from a callable function.
        
        Args:
            field_name: Field component name (e.g., 'E_z', 'H_x')
            func: Callable function that defines the field distribution
            dt: Time step size
            times: If provided as integer, returns time series of true solutions
        
        Returns:
            If times is None: field array
            If times is provided: tuple of (field_array, time_series_data)
            
        Raises:
            ValueError: If field_name format is invalid
        """
        TD = self.TD
        device = self.device

        kind, comp = field_name.split("_")
        if kind not in {"E", "H"}:
            raise ValueError("field_name must start with 'E' or 'H'")

        base_t = 0.0 if kind == "E" else -0.5 * dt

        node = self.node.reshape(self.nx + 1, self.ny + 1, getattr(self, "nz", 0) + 1, self.TD).squeeze()

        if TD == 2:
            if kind == "E" and comp == "z":
                coords = node
            elif kind == "H" and comp == "x":
                coords = self.edgey_coords
            elif kind == "H" and comp == "y":
                coords = self.edgex_coords
            else:
                raise ValueError("2D supports only 'E_z', 'H_x', 'H_y'")
            X, Y = coords[..., 0], coords[..., 1]
            try:
                arr = func(X, Y, base_t)
            except TypeError:
                # func may not accept time argument
                arr = func(X, Y)
        else:
            def avg(a, axis):
                sl = [slice(None)] * a.ndim
                fwd = sl.copy()
                fwd[axis] = slice(1, None)
                bwd = sl.copy()
                bwd[axis] = slice(None, -1)
                return 0.5 * (a[tuple(fwd)] + a[tuple(bwd)])

            axis = "xyz".index(comp)
            if kind == "E":
                coords = avg(node, axis)
            else:
                coords = node
                for ax in sorted({0, 1, 2} - {axis}, reverse=True):
                    coords = avg(coords, ax)
            X, Y, Z = coords[..., 0], coords[..., 1], coords[..., 2]
            try:
                arr = func(X, Y, Z, base_t)
            except TypeError:
                arr = func(X, Y, Z)

        # If times is provided, compute time series of true solutions
        if times is not None:
            if isinstance(times, int):
                t_pts = base_t + dt * bm.arange(times + 1, device=device)
            else:
                t_pts = bm.asarray(list(times), dtype=float, device=device)

            if TD == 2:
                data = bm.stack([func(X, Y, t) for t in t_pts], axis=0)
            else:
                data = bm.stack([func(X, Y, Z, t) for t in t_pts], axis=0)

            return arr, data

        return arr
    
    def init_field_matrix(self, type: Optional[str] = None, axis: Optional[str] = None) -> Any:
        """
        Initialize field matrix for given field type and axis.
        
        Args:
            type: Field type ('E' or 'H')
            axis: Field component axis ('x', 'y', 'z')
            
        Returns:
            Zero-initialized field matrix of appropriate shape
            
        Raises:
            ValueError: If parameters are invalid or dimension unsupported
        """
        dim = self.TD
        if dim == 2:
            if type == "E":
                return self.get_field_matrix(etype="node", axis=axis)
            elif type == "H":
                return self.get_field_matrix(etype="edge", axis=axis)
        elif dim == 3:
            if type == "E":
                return self.get_field_matrix(etype="edge", axis=axis)
            elif type == "H":
                return self.get_field_matrix(etype="face", axis=axis)
        raise ValueError(f"Invalid type='{type}' or unsupported mesh dimension {dim}")

    def _init_fields_dict(self, field: str, comps: List[str], num_frames: int, axis_type: bool = True):
        """
        Initialize field dictionaries for simulation.
        
        Args:
            field: Field type ('E' or 'H')
            comps: Field components to initialize
            num_frames: Number of frames for time series data
            axis_type: Whether to use axis-type field storage
            
        Returns:
            Tuple of (fields_dict, data_dict)
        """
        fields = {}
        data = {}

        def _copy_array(arr):
            """Use bm.copy when available; fallback to manual zero+assign."""
            try:
                return bm.copy(arr)
            except Exception:
                # Conservative fallback: allocate same-shape zero and assign
                dst = bm.zeros(arr.shape, dtype=arr.dtype, device=self.device)
                dst[...] = arr
                return dst

        for comp in comps:
            # Determine base initial array
            base = None
            if field == "E":
                existing = getattr(self, "E", {}).get(comp, None)
                if existing is not None:
                    base = existing
            elif field == "H":
                existing = getattr(self, "H", {}).get(comp, None)
                if existing is not None:
                    base = existing

            if base is None:
                base = self.init_field_matrix(field, comp)

            # Build fields entry
            if axis_type:
                f0 = base
                f1 = _copy_array(f0)
                fields[comp] = (f0, f1)
                ref = f0
            else:
                fields[comp] = base
                ref = base

            # Frame buffer
            if num_frames:
                buf_shape = (num_frames, *ref.shape)
                buf = bm.zeros(buf_shape, dtype=ref.dtype, device=self.device)
                buf[0] = ref
                data[comp] = buf

        return fields, data