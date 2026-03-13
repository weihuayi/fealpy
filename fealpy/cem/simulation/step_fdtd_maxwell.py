from typing import Optional, Union, Callable, Dict, Any, Tuple
from fealpy.backend import backend_manager as bm
import math

# Physical constants
c0 = 299792458.0
mu0 = 4.0 * math.pi * 1e-7
eta0: float = mu0 * c0


class StepFDTDMaxwell:
    """
    Finite-Difference Time-Domain (FDTD) solver for Maxwell's equations.
    
    This class implements the Yee grid-based FDTD method for solving
    time-domain electromagnetic wave propagation problems.
    
    Attributes:
        yee: Yee grid object
        mesh: Computational mesh
        device: Computation device (CPU/GPU)
        dtype: Data type for computations
        boundary: Boundary condition type ('PEC' or 'UPML')
        pml_width: Width of PML layers in grid points
        pml_m: PML grading parameter
        nx, ny, nz: Grid dimensions
        TD: Topological dimension (2D or 3D)
        h: Grid spacing
        R: Courant number
        dt: Time step size
    """

    def __init__(self,
                 yee,
                 R: Optional[float] = None,
                 dt: Optional[float] = None,
                 boundary: str = "PEC",
                 pml_width: int = 8,
                 pml_m: float = 1.0,
                 eps: Union[float, Any] = 1.0,  # Supports scalar or matrix
                 mu: Union[float, Any] = 1.0,   # Supports scalar or matrix
                 dtype: str = "float64"):
        """
        Initialize FDTD solver.

        Args:
            yee: Yee grid object
            R: Courant number (if None, calculated automatically)
            dt: Time step size (if None, calculated automatically)
            boundary: Boundary condition type ('PEC' or 'UPML')
            pml_width: Width of PML layers in grid points
            pml_m: PML grading parameter
            eps: Relative permittivity (scalar or matrix)
            mu: Relative permeability (scalar or matrix)
            dtype: Data type for computations
        """
        self.yee = yee
        self.mesh = getattr(yee, "mesh", yee)
        self.device = getattr(self.mesh, "device", None)
        self.dtype = dtype
        self.boundary = (boundary or "PEC").upper()
        self.pml_width = int(pml_width)
        self.pml_m = float(pml_m)
        self._register_update_methods()

        # Grid sizes & spacing (try to read from mesh/yee reliably)
        self.nx = int(getattr(self.mesh, "nx"))
        self.ny = int(getattr(self.mesh, "ny"))
        self.nz = int(getattr(self.mesh, "nz", 0))
        self.TD = int(getattr(self.mesh, "TD", 2 if self.nz == 0 else 3))

        self.h = self.mesh.h[0]

        self.Eps = self.init_perm_matrix(eps)
        self.Mu = self.init_perm_matrix(mu)
        self.get_perm_matrix()

        # Determine time step parameters
        if R is not None and dt is not None:
            self.R = R
            self.dt = dt
        elif R is not None:
            # Only R provided, calculate dt from R
            self.R = R
            self.dt = self.h * self.R / c0
        elif dt is not None:
            # Only dt provided, calculate R from dt
            self.dt = dt
            self.R = c0 * self.dt / self.h
        else:
            # Neither provided, use default CFL condition
            if self.TD == 2:
                default_R = 1 / math.sqrt(2)  # 2D CFL condition: R <= 1/√2 ≈ 0.707
            else:
                default_R = 1 / math.sqrt(3)  # 3D CFL condition: R <= 1/√3 ≈ 0.577
            
            self.R = 0.99 * default_R
            self.dt = self.h * self.R / c0
        
            import warnings
            warnings.warn(f"Neither R nor dt provided. Using default R={default_R} and dt={self.dt:.2e}")

        self._check_stability()

    def _check_stability(self) -> None:
        """Check CFL stability condition."""
        if self.TD == 2:
            max_R = 1.0 / math.sqrt(2)  # 2D CFL condition
        else:
            max_R = 1.0 / math.sqrt(3)  # 3D CFL condition
            
        if self.R > max_R:
            import warnings
            warnings.warn(f"CFL stability condition may be violated: R={self.R:.3f} > max_R={max_R:.3f}")

    def init_perm_matrix(self, perm: Union[float, int, Any]) -> Any:
        """
        Initialize permittivity/permeability matrix.
        
        Args:
            perm: Permittivity or permeability value (scalar or matrix)
            
        Returns:
            Matrix with appropriate shape and values
            
        Raises:
            ValueError: If matrix shape mismatch
        """
        node_shape = self.yee.get_field_matrix("node").shape
        if isinstance(perm, (float, int)):
            return bm.full(node_shape, perm, device=self.device)
        elif hasattr(perm, 'shape') and perm.shape == node_shape:
            return bm.asarray(perm, device=self.device)
        else:
            raise ValueError(f"perm array shape mismatch. Expected {node_shape}, got {getattr(perm, 'shape', 'unknown')}")

    def get_perm_matrix(self) -> None:
        """
        Calculate and store inverse permittivity and permeability matrices
        for update equations.
        """
        if self.TD == 2:
            # Magnetic permeability staggered at x- and y- edges
            Mu = self.Mu
            Mux = 0.5 * (Mu[:, :-1] + Mu[:, 1:])    # Average in y-direction
            Muy = 0.5 * (Mu[:-1, :] + Mu[1:, :])    # Average in x-direction

            # Electric permittivity is node-centered (z-component is unchanged)
            epsz = self.Eps

            # Store inverses for update equations
            self.inv_Mu_x = 1.0 / Mux
            self.inv_Mu_y = 1.0 / Muy
            self.inv_Eps_z = 1.0 / epsz

        else:
            # 3D: Compute face-centered permittivity
            Eps = self.Eps
            # Permittivity on faces perpendicular to x, y, z
            self.inv_Eps_x = 1.0 / (0.5 * (Eps[:-1, :, :] + Eps[1:, :, :]))
            self.inv_Eps_y = 1.0 / (0.5 * (Eps[:, :-1, :] + Eps[:, 1:, :]))
            self.inv_Eps_z = 1.0 / (0.5 * (Eps[:, :, :-1] + Eps[:, :, 1:]))

            # 3D: Compute edge-centered permeability by averaging the four nodes around each edge
            Mu = self.Mu
            Mux = 0.25 * (
                Mu[:, :-1, :-1] + Mu[:, 1:, :-1] +
                Mu[:, :-1, 1:] + Mu[:, 1:, 1:]
            )
            Muy = 0.25 * (
                Mu[:-1, :, :-1] + Mu[1:, :, :-1] +
                Mu[:-1, :, 1:] + Mu[1:, :, 1:]
            )
            Muz = 0.25 * (
                Mu[:-1, :-1, :] + Mu[1:, :-1, :] + 
                Mu[:-1, 1:, :] + Mu[1:, 1:, :]
            )

            # Store inverses for magnetic update
            self.inv_Mu_x = 1.0 / Mux
            self.inv_Mu_y = 1.0 / Muy
            self.inv_Mu_z = 1.0 / Muz

    # ============================ Update Functions ============================

    # ============================ UPML Boundary Conditions ============================
    def create_sigma_function(self, dim: int, ng: int, m: float, R0: float = 0.001) -> Callable:
        """
        Create sigma function for UPML boundary conditions.
        
        Args:
            dim: Dimension index (0=x, 1=y, 2=z)
            ng: Number of PML layers
            m: Grading parameter
            R0: Reference reflection coefficient
            
        Returns:
            Function that calculates sigma values for specified dimension
        """
        sup = bm.max(self.mesh.node, axis=0)
        inf = bm.min(self.mesh.node, axis=0)

        sigma = 1 / self.h

        # Precompute invariants related to specified dimension
        pml_len = (sup[dim] - inf[dim]) * ng / (self.mesh.extent[dim*2+1] - self.mesh.extent[dim*2])

        l0 = inf[dim] + pml_len
        l1 = sup[dim] - pml_len
        
        def sigma_func(p):
            coord = p[..., dim]

            # Use bm.where for vectorized computation
            return bm.where(coord < l0, sigma * ((l0 - coord) / pml_len) ** m,
                            bm.where(coord > l1, sigma * ((coord - l1) / pml_len) ** m, 0))
        
        return sigma_func
    
    # ---------------- Registry (map boundary+dim -> runner builder) ----------------
    def _register_update_methods(self) -> None:
        """Register update methods for different boundary conditions and dimensions."""
        self._update_methods = {
            ('UPML', 2): self._one_step_upml_2d,
            ('UPML', 3): self._one_step_upml_3d,
            ('PEC',  2): self._one_step_pec_2d,
            ('PEC',  3): self._one_step_pec_3d,
        }

    # ---------------- Modify boundary parameters and invalidate precomputation ----------------
    def set_boundary_params(self, 
                          boundary: Optional[str] = None,
                          pml_width: Optional[int] = None,
                          pml_m: Optional[float] = None) -> None:
        """
        Modify boundary parameters and clear boundary-related precomputation (e.g., PML coefficients).
        
        Args:
            boundary: New boundary condition type
            pml_width: New PML width in grid points
            pml_m: New PML grading parameter
        """
        boundary_changed = False
        
        if boundary is not None:
            new_boundary = boundary.upper()
            if new_boundary != self.boundary:
                self.boundary = new_boundary
                boundary_changed = True
        
        if pml_width is not None:
            self.pml_width = int(pml_width)
        if pml_m is not None:
            self.pml_m = float(pml_m)
        
        # Clear precomputation if boundary condition changed or PML parameters changed
        if boundary_changed or pml_width is not None or pml_m is not None:
            if hasattr(self, "_upml_prepared"):
                delattr(self, "_upml_prepared")

    def prepare_upml_2d(self) -> None:
        """
        Precompute and cache sigma and coefficients for UPML 2D (done only once).
        _one_step_upml_2d will use these caches for high performance within single step.
        """
        if getattr(self, "_upml_prepared", False):
            return

        R = float(self.R)
        h = float(self.h)
        m = float(self.pml_m)
        ng = int(self.pml_width)

        sigma_x = self.create_sigma_function(0, ng, m)
        sigma_y = self.create_sigma_function(1, ng, m)

        sx0 = self.yee.interpolation(sigma_x, intertype='node')    # Node-centered
        sy0 = self.yee.interpolation(sigma_y, intertype='node')

        sx1 = self.yee.interpolation(sigma_x, intertype='edgex')   # Edgex
        sy1 = self.yee.interpolation(sigma_y, intertype='edgex')

        sx2 = self.yee.interpolation(sigma_x, intertype='edgey')   # Edgey
        sy2 = self.yee.interpolation(sigma_y, intertype='edgey')

        # Cache coefficients for update equations
        self.c1 = (2 - sy2 * R * h) / (2 + sy2 * R * h)
        self.c2 = 2 * R / (2 + sy2 * R * h)

        self.c3 = (2 - sx1 * R * h) / (2 + sx1 * R * h)
        self.c4 = 2 * R / (2 + sx1 * R * h)

        self.c5 = (2 + sx2 * R * h) / 2
        self.c6 = (2 - sx2 * R * h) / 2

        self.c7 = (2 + sy1 * R * h) / 2
        self.c8 = (2 - sy1 * R * h) / 2

        self.c9  = (2 - sx0[1:-1, 1:-1] * R * h) / (2 + sx0[1:-1, 1:-1] * R * h)
        self.c10 = 2 * R / (2 + sx0[1:-1, 1:-1] * R * h)

        self.c11 = (2 - sy0[1:-1, 1:-1] * R * h) / (2 + sy0[1:-1, 1:-1] * R * h)
        self.c12 = 2 / (2 + sy0[1:-1, 1:-1] * R * h)

        self._upml_prepared = True

    def prepare_upml_3d(self) -> None:
        """Precompute and cache coefficients for UPML 3D."""
        if getattr(self, "_upml_prepared", False):
            return
        
        R = float(self.R)
        h = float(self.h)
        m = float(self.pml_m)
        ng = int(self.pml_width)

        sigma_x = self.create_sigma_function(0, ng, m)
        sigma_y = self.create_sigma_function(1, ng, m)
        sigma_z = self.create_sigma_function(2, ng, m)
        
        # Interpolate sigma values to different grid locations
        sx0 = self.yee.interpolation(sigma_x, intertype='facex')
        sy0 = self.yee.interpolation(sigma_y, intertype='facex')
        sz0 = self.yee.interpolation(sigma_z, intertype='facex')

        sx1 = self.yee.interpolation(sigma_x, intertype='facey')
        sy1 = self.yee.interpolation(sigma_y, intertype='facey')
        sz1 = self.yee.interpolation(sigma_z, intertype='facey')

        sx2 = self.yee.interpolation(sigma_x, intertype='facez')
        sy2 = self.yee.interpolation(sigma_y, intertype='facez')
        sz2 = self.yee.interpolation(sigma_z, intertype='facez')

        sx3 = self.yee.interpolation(sigma_x, intertype='edgex')
        sy3 = self.yee.interpolation(sigma_y, intertype='edgex')
        sz3 = self.yee.interpolation(sigma_z, intertype='edgex')

        sx4 = self.yee.interpolation(sigma_x, intertype='edgey')
        sy4 = self.yee.interpolation(sigma_y, intertype='edgey')
        sz4 = self.yee.interpolation(sigma_z, intertype='edgey')

        sx5 = self.yee.interpolation(sigma_x, intertype='edgez')
        sy5 = self.yee.interpolation(sigma_y, intertype='edgez')
        sz5 = self.yee.interpolation(sigma_z, intertype='edgez')
    
        # Cache coefficients for 3D UPML update equations
        self.c1 = (2 - sz0 * R * h) / (2 + sz0 * R * h)
        self.c2 = 2 * R / (2 + sz0 * R * h)

        self.c3 = (2 - sx1 * R * h) / (2 + sx1 * R * h)
        self.c4 = 2 * R / (2 + sx1 * R * h)

        self.c5 = (2 - sy2 * R * h) / (2 + sy2 * R * h)
        self.c6 = 2 * R / (2 + sy2 * R * h)

        self.c7 = (2 - sy0 * R * h) / (2 + sy0 * R * h)
        self.c8 = (2 + sx0 * R * h) / (2 + sy0 * R * h)
        self.c9 = (2 - sx0 * R * h) / (2 + sy0 * R * h)

        self.c10 = (2 - sz1 * R * h) / (2 + sz1 * R * h)
        self.c11 = (2 + sy1 * R * h) / (2 + sz1 * R * h)
        self.c12 = (2 - sy1 * R * h) / (2 + sz1 * R * h)

        self.c13 = (2 - sx2 * R * h) / (2 + sx2 * R * h)
        self.c14 = (2 + sz2 * R * h) / (2 + sx2 * R * h)
        self.c15 = (2 - sz2 * R * h) / (2 + sx2 * R * h)

        self.c16 = (2 - sz3[:, 1:-1, 1:-1] * R * h) / (2 + sz3[:, 1:-1, 1:-1] * R * h)
        self.c17 = 2 * R / (2 + sz3[:, 1:-1, 1:-1] * R * h)

        self.c18 = (2 - sx4[1:-1, :, 1:-1] * R * h) / (2 + sx4[1:-1, :, 1:-1] * R * h)
        self.c19 = 2 * R / (2 + sx4[1:-1, :, 1:-1] * R * h)

        self.c20 = (2 - sy5[1:-1, 1:-1, :] * R * h) / (2 + sy5[1:-1, 1:-1, :] * R * h)
        self.c21 = 2 * R / (2 + sy5[1:-1, 1:-1, :] * R * h)

        self.c22 = (2 - sy3[:, 1:-1, 1:-1] * R * h) / (2 + sy3[:, 1:-1, 1:-1] * R * h)
        self.c23 = (2 + sx3[:, 1:-1, 1:-1] * R * h) / (2 + sy3[:, 1:-1, 1:-1] * R * h)
        self.c24 = (2 - sx3[:, 1:-1, 1:-1] * R * h) / (2 + sy3[:, 1:-1, 1:-1] * R * h)

        self.c25 = (2 - sz4[1:-1, :, 1:-1] * R * h) / (2 + sz4[1:-1, :, 1:-1] * R * h)
        self.c26 = (2 + sy4[1:-1, :, 1:-1] * R * h) / (2 + sz4[1:-1, :, 1:-1] * R * h)
        self.c27 = (2 - sy4[1:-1, :, 1:-1] * R * h) / (2 + sz4[1:-1, :, 1:-1] * R * h)

        self.c28 = (2 - sx5[1:-1, 1:-1, :] * R * h) / (2 + sx5[1:-1, 1:-1, :] * R * h)
        self.c29 = (2 + sz5[1:-1, 1:-1, :] * R * h) / (2 + sx5[1:-1, 1:-1, :] * R * h)
        self.c30 = (2 - sz5[1:-1, 1:-1, :] * R * h) / (2 + sx5[1:-1, 1:-1, :] * R * h)

        self._upml_prepared = True

    # ---------------- One-step implementation (UPML 2D) ----------------
    def _one_step_upml_2d(self, 
                         E: Dict[str, Any], 
                         H: Dict[str, Any], 
                         D: Dict[str, Any], 
                         B: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Perform one time step update for 2D UPML boundary conditions.
        
        Args:
            E: Electric field components
            H: Magnetic field components  
            D: Electric displacement field components
            B: Magnetic flux density components
            
        Returns:
            Updated E, H, D, B fields
        """
        if not getattr(self, "_upml_prepared", False):
            self.prepare_upml_2d()

        # Retrieve cached coefficients and inverse arrays
        c1, c2, c3, c4 = self.c1, self.c2, self.c3, self.c4
        c5, c6, c7, c8 = self.c5, self.c6, self.c7, self.c8
        c9, c10, c11, c12 = self.c9, self.c10, self.c11, self.c12

        inv_eps_z = self.inv_Eps_z
        inv_Mu_x = self.inv_Mu_x
        inv_Mu_y = self.inv_Mu_y

        # --- Update B (magnetic auxiliary) ---
        B_old_x = bm.asarray(B['x']).copy()
        B_old_y = bm.asarray(B['y']).copy()

        B['x'] = c1 * B_old_x - c2 * (E['z'][:, 1:] - E['z'][:, :-1]) 
        B['y'] = c3 * B_old_y + c4 * (E['z'][1:, :] - E['z'][:-1, :])

        # --- Update H from B ---
        H['x'] += inv_Mu_x * (c5 * B['x'] - c6 * B_old_x)
        H['y'] += inv_Mu_y * (c7 * B['y'] - c8 * B_old_y)

        # --- Update D (interior) ---
        D_old_z = bm.asarray(D['z']).copy()
        curlH = (H['y'][1:, 1:-1] - H['y'][:-1, 1:-1]) - (H['x'][1:-1, 1:] - H['x'][1:-1, :-1])
        D['z'][1:-1, 1:-1] = c9 * D_old_z[1:-1, 1:-1] + c10 * curlH

        # --- Update E from D (interior) ---
        E_old_z = bm.asarray(E['z']).copy()
        E['z'][1:-1, 1:-1] = c11 * E_old_z[1:-1, 1:-1] + c12 * inv_eps_z[1:-1, 1:-1] * (D['z'][1:-1, 1:-1] - D_old_z[1:-1, 1:-1])

        return E, H, D, B

    def _one_step_upml_3d(self,
                         E: Dict[str, Any],
                         H: Dict[str, Any], 
                         D: Dict[str, Any],
                         B: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Perform one time step update for 3D UPML boundary conditions.
        
        Args:
            E: Electric field components
            H: Magnetic field components  
            D: Electric displacement field components
            B: Magnetic flux density components
            
        Returns:
            Updated E, H, D, B fields
        """
        if not getattr(self, "_upml_prepared", False):
            self.prepare_upml_3d()

        # Retrieve cached coefficients
        c1,  c2,  c3,  c4,  c5  = self.c1,  self.c2,  self.c3,  self.c4,  self.c5
        c6,  c7,  c8,  c9,  c10 = self.c6,  self.c7,  self.c8,  self.c9,  self.c10
        c11, c12, c13, c14, c15 = self.c11, self.c12, self.c13, self.c14, self.c15
        c16, c17, c18, c19, c20 = self.c16, self.c17, self.c18, self.c19, self.c20
        c21, c22, c23, c24, c25 = self.c21, self.c22, self.c23, self.c24, self.c25
        c26, c27, c28, c29, c30 = self.c26, self.c27, self.c28, self.c29, self.c30

        inv_Eps_x = self.inv_Eps_x
        inv_Eps_y = self.inv_Eps_y
        inv_Eps_z = self.inv_Eps_z
        inv_Mu_x = self.inv_Mu_x
        inv_Mu_y = self.inv_Mu_y
        inv_Mu_z = self.inv_Mu_z
        
        # Create copies of old field values
        E_old = {comp: E[comp].copy() for comp in ['x', 'y', 'z']}
        H_old = {comp: H[comp].copy() for comp in ['x', 'y', 'z']}
        D_old = {comp: D[comp].copy() for comp in ['x', 'y', 'z']}
        B_old = {comp: B[comp].copy() for comp in ['x', 'y', 'z']}
        
        # B update - using old E values
        B['x'] = c1 * B_old['x'] - c2 * (
            (E_old['z'][:, 1:, :] - E_old['z'][:, :-1, :]) - 
            (E_old['y'][:, :, 1:] - E_old['y'][:, :, :-1])
        )
        B['y'] = c3 * B_old['y'] - c4 * (
            (E_old['x'][:, :, 1:] - E_old['x'][:, :, :-1]) - 
            (E_old['z'][1:, :, :] - E_old['z'][:-1, :, :])
        )
        B['z'] = c5 * B_old['z'] - c6 * (
            (E_old['y'][1:, :, :] - E_old['y'][:-1, :, :]) - 
            (E_old['x'][:, 1:, :] - E_old['x'][:, :-1, :])
        )
        
        # H update - using old H values and new/old B values
        H['x'] = c7 * H_old['x'] + inv_Mu_x * (c8 * B['x'] - c9 * B_old['x'])
        H['y'] = c10 * H_old['y'] + inv_Mu_y * (c11 * B['y'] - c12 * B_old['y'])
        H['z'] = c13 * H_old['z'] + inv_Mu_z * (c14 * B['z'] - c15 * B_old['z'])
        
        # D update - using old D values and new H values (with boundary handling)
        D['x'][:, 1:-1, 1:-1] = c16 * D_old['x'][:, 1:-1, 1:-1] + c17 * (
            (H['z'][:, 1:, 1:-1] - H['z'][:, :-1, 1:-1]) - 
            (H['y'][:, 1:-1, 1:] - H['y'][:, 1:-1, :-1])
        )
        D['y'][1:-1, :, 1:-1] = c18 * D_old['y'][1:-1, :, 1:-1] + c19 * (
            (H['x'][1:-1, :, 1:] - H['x'][1:-1, :, :-1]) - 
            (H['z'][1:, :, 1:-1] - H['z'][:-1, :, 1:-1])
        )
        D['z'][1:-1, 1:-1, :] = c20 * D_old['z'][1:-1, 1:-1, :] + c21 * (
            (H['y'][1:, 1:-1, :] - H['y'][:-1, 1:-1, :]) - 
            (H['x'][1:-1, 1:, :] - H['x'][1:-1, :-1, :])
        )
        
        # E update - using old E values and new/old D values (with boundary handling)
        E['x'][:, 1:-1, 1:-1] = c22 * E_old['x'][:, 1:-1, 1:-1] + inv_Eps_x[:, 1:-1, 1:-1] * (
            c23 * D['x'][:, 1:-1, 1:-1] - c24 * D_old['x'][:, 1:-1, 1:-1]
        )
        E['y'][1:-1, :, 1:-1] = c25 * E_old['y'][1:-1, :, 1:-1] + inv_Eps_y[1:-1, :, 1:-1] * (
            c26 * D['y'][1:-1, :, 1:-1] - c27 * D_old['y'][1:-1, :, 1:-1]
        )
        E['z'][1:-1, 1:-1, :] = c28 * E_old['z'][1:-1, 1:-1, :] + inv_Eps_z[1:-1, 1:-1, :] * (
            c29 * D['z'][1:-1, 1:-1, :] - c30 * D_old['z'][1:-1, 1:-1, :]
        )

        return E, H, D, B

    # ---------------- One-step implementation (PEC 2D) ----------------
    def _one_step_pec_2d(self, E: Dict[str, Any], H: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """
        Perform one time step update for 2D PEC boundary conditions.
        
        Args:
            E: Electric field components
            H: Magnetic field components
            
        Returns:
            Updated E and H fields
        """
        H['x'] -= self.R * self.inv_Mu_x * (E['z'][:, 1:] - E['z'][:, 0:-1])
        H['y'] += self.R * self.inv_Mu_y * (E['z'][1:, :] - E['z'][0:-1, :])

        E['z'][1:-1, 1:-1] += self.R * self.inv_Eps_z[1:-1, 1:-1] * (
            H['y'][1:, 1:-1] - H['y'][0:-1, 1:-1] - 
            H['x'][1:-1, 1:] + H['x'][1:-1, 0:-1]
        )

        return E, H
    
    def _one_step_pec_3d(self, E: Dict[str, Any], H: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """
        Perform one time step update for 3D PEC boundary conditions.
        
        Args:
            E: Electric field components
            H: Magnetic field components
            
        Returns:
            Updated E and H fields
        """
        H['x'] -= self.R * self.inv_Mu_x * (
            (E['z'][:, 1:, :] - E['z'][:, :-1, :]) - 
            (E['y'][:, :, 1:] - E['y'][:, :, :-1])
        )
                
        H['y'] -= self.R * self.inv_Mu_y * (
            (E['x'][:, :, 1:] - E['x'][:, :, :-1]) - 
            (E['z'][1:, :, :] - E['z'][:-1, :, :])
        )
        
        H['z'] -= self.R * self.inv_Mu_z * (
            (E['y'][1:, :, :] - E['y'][:-1, :, :]) - 
            (E['x'][:, 1:, :] - E['x'][:, :-1, :])
        )

        E['x'][:, 1:-1, 1:-1] += self.R * self.inv_Eps_x[:, 1:-1, 1:-1] * (
            (H['z'][:, 1:, 1:-1] - H['z'][:, :-1, 1:-1]) -
            (H['y'][:, 1:-1, 1:] - H['y'][:, 1:-1, :-1])
        )

        E['y'][1:-1, :, 1:-1] += self.R * self.inv_Eps_y[1:-1, :, 1:-1] * (
            (H['x'][1:-1, :, 1:] - H['x'][1:-1, :, :-1]) -
            (H['z'][1:, :, 1:-1] - H['z'][:-1, :, 1:-1])
        )

        E['z'][1:-1, 1:-1, :] += self.R * self.inv_Eps_z[1:-1, 1:-1, :] * (
            (H['y'][1:, 1:-1, :] - H['y'][:-1, 1:-1, :]) -
            (H['x'][1:-1, 1:, :] - H['x'][1:-1, :-1, :])
        )

        return E, H

    def update(self, 
               E: Dict[str, Any], 
               H: Dict[str, Any], 
               D: Optional[Dict[str, Any]] = None, 
               B: Optional[Dict[str, Any]] = None) -> Tuple[Dict, ...]:
        """
        Perform one time step update: automatically select corresponding update method
        based on current boundary condition and dimension.
        
        Args:
            E: Electric field components
            H: Magnetic field components
            D: Electric displacement field components (required for UPML)
            B: Magnetic flux density components (required for UPML)
            
        Returns:
            Updated field components
            
        Raises:
            ValueError: If no update method available or missing required parameters
        """
        # Dynamically select update method
        method_key = (self.boundary, self.TD)
        
        if method_key not in self._update_methods:
            raise ValueError(f"No update method available for {self.boundary} boundary in {self.TD}D")
        
        runner = self._update_methods[method_key]
        
        # Check parameters and call corresponding runner based on boundary condition
        if self.boundary == "UPML":
            if D is None or B is None:
                raise ValueError("UPML runner requires D and B dictionaries; pass them to update().")
            return runner(E, H, D, B)
        else:
            # PEC boundary only needs E and H
            return runner(E, H)