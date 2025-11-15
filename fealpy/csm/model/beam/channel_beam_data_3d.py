from fealpy.typing import Tuple, TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.mesh import EdgeMesh


class ChannelBeamData3D:
    """ Channel beam data model for 3D beam analysis. 
    This class provides geometric and material parameters for a channel beam
    with specific cross-sectional properties.
    
    Parameters:
        mu_y (float): Ratio of maximum to average shear stress for y-direction shear, 
            default is 2.44.
        mu_z (float): Ratio of maximum to average shear stress for z-direction shear,
            default is 2.38.
    """
    def __init__(self, mu_y: float=2.44, mu_z: float=2.38):
        
        self.FSY = mu_y
        self.FSZ = mu_z
        
        self.GD = self.geo_dimension()
        
        self.L = self.length()
        self.Ax, self.Ay, self.Az = self.cross_section()
        self.Ix, self.Iy, self.Iz = self.inertia()
        
        self.e_z = 0.0148  # Shear center offset in z-direction (m)
        self.W_t = 8.64e-7  # Torsional section modulus
        self.stress_points = self.get_stress_points()
        
        self.dofs_per_node = 6
        self.mesh = self.init_mesh() 
    
    def __str__(self) -> str:
        """Returns a formatted multi-line string summarizing the configuration of the 3D Channel beam data.
        
        Returns:
            str: A multi-line string showing key beam parameters, geometry, and mesh info.
        """
        s = f"{self.__class__.__name__}(\n"
        s += "  === Geometry & Mesh ===\n"
        s += f"  Mesh Type             : {self.mesh.__class__.__name__}\n"
        s += f"  Number of Nodes       : {self.mesh.number_of_nodes()}\n"
        s += f"  Number of Elements    : {self.mesh.number_of_cells()}\n"
        s += f"  Geo Dimension         : {self.GD}\n"
        s += f"  Beam Length           : {self.L} m\n"
        s += "\n  === Cross-Sectional Properties ===\n"
        s += f"  Area (A)              : {self.A} m²\n"
        s += f"  Torsional Constant (J): {self.Ix} m⁴\n"
        s += f"  Moment of Inertia (Iy): {self.Iy} m⁴\n"
        s += f"  Moment of Inertia (Iz): {self.Iz} m⁴\n"
        s += f"  Shear Center Offset   : {self.e_z} m\n"
        s += f"  Torsional Modulus (Wt): {self.W_t} m³\n"
        s += "\n  === Shear Factors ===\n"
        s += f"  mu_y                  : {self.FSY}\n"
        s += f"  mu_z                  : {self.FSZ}\n"
        s += ")"
        return s
    
    def geo_dimension(self) -> int:
        """Get the geometric dimension of the beam."""
        return 3

    def length(self) -> float:
        """Beam length."""
        return 1.0

    def cross_section(self) -> float:
        """ Get the beam cross-sectional area."""
        Ax = 4.90e-4
        Ay = 4.90e-4
        Az = 4.90e-4
        return  Ax, Ay, Az

    def inertia(self) -> Tuple[float, float, float]:
        """Get the beam moments of inertia.
        
        Returns:
            A tuple containing:
            - Ix (J): Torsional constant in m⁴
            - Iy: Weak axis moment of inertia in m⁴
            - Iz: Strong axis moment of inertia in m⁴
        """
        Ix = 5.18e-9
        Iy = 2.77e-8
        Iz = 1.69e-7
        return Ix, Iy, Iz

    def get_stress_points(self) -> TensorLike:
        """ Get the stress calculation points at the outermost corners of the 
        cross-section in local coordinates.
        
        Returns:
            Array of shape (4, 2) containing (y, z) coordinates of the 
            four corner points in meters.
            
        Note:
            y-axis: horizontal direction
            z-axis: vertical direction
        """
        # Define stress points in the cross-section (y, z)
        points = bm.array([
            [-0.025, -0.0164],  # Point 1: (y1, z1)
            [0.025, -0.0164],   # Point 2: (y2, z2)
            [0.025, 0.0086],    # Point 3: (y3, z3)
            [-0.025, 0.0086]    # Point 4: (y4, z4)
        ], dtype=bm.float64)
        return points
    
    def init_mesh(self, n: int=10):
        """Creates a 1D mesh along the beam length for finite element analysis.

        Parameters:
            n (int): Number of elements along the beam length, default is 10.
        """
        nodes = bm.linspace(0, self.L, n + 1)
        node = bm.zeros((n + 1, 3), dtype=bm.float64)
        node[:, 0] = nodes  # x-coordinates along beam length
        
        cell = bm.zeros((n, 2), dtype=bm.int32)
        cell[:, 0] = bm.arange(n)
        cell[:, 1] = bm.arange(1, n + 1)
        
        return EdgeMesh(node, cell)
    
    def tip_load(self,  load_case: int=1) -> TensorLike:
        """ Get the concentrated load at the tip of the beam.

        Parameters:
            load_case (int, optional): load case number (1 or 2), default is 1.

        Notes:
            Load case 1: Three forces and one torque at the tip.
                - Axial force: Fx = 10 N
                - Transverse forces: Fy = 50 N, Fz = 100 N
                - Torque: Mx = -10 N·m
            Load case 2: Only gravity (no concentrated tip load).
        """
        load = bm.zeros(6, dtype=bm.float64)
        if load_case == 1:
            load[0] = 10.0    # Fx
            load[1] = 50.0    # Fy
            load[2] = 100.0   # Fz
            load[3] = -10.0   # Mx
        return load
    
    def dirichlet(self, points: TensorLike) -> TensorLike:
        """Get Dirichlet boundary condition values.

        Parameters:
            points (TensorLike): Spatial points where Dirichlet BCs are evaluated.

        Notes:
            One end of the beam is fixed (clamped), resulting in zero 
        displacements and rotations.
        """
        shape = points.shape[:-1] + (6,)
        return bm.zeros(shape, dtype=bm.float64)
     
    def is_dirichlet_boundary(self, points: TensorLike) -> TensorLike:
        """Identify nodes on the Dirichlet boundary.

        Parameters:
            points (TensorLike): Spatial coordinates of nodes, shape (NN, GD).

        Notes:
            The fixed end is at x = 0.
        """
        return bm.abs(points[..., 0]) < 1e-12
    
    def dirichlet_dof(self) -> TensorLike:
        """Get the indices of degrees of freedom with Dirichlet boundary conditions.

        Notes:
            Fixed end conditions are applied at x = 0 (first node).
        """
        mesh = self.mesh
        node = mesh.entity('node')
        
        is_fixed = self.is_dirichlet_boundary(node)
        fixed_node_indices = bm.where(is_fixed)[0]
        
        dofs = []
        for node_idx in fixed_node_indices:
            for i in range(self.dofs_per_node):
                dofs.append(node_idx * self.dofs_per_node + i)
                
        return bm.array(dofs, dtype=bm.int32)