from fealpy.typing import Tuple, TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesh import EdgeMesh
from ...mesh import TrussTower


class TrussTowerData3D:
    """3D Truss Tower data structure."""
    
    def __init__(self,
                 dov=0.015, div=0.010,    # Vertical rod outer / inner diameter (m)
                 doo=0.010, dio=0.007     # Other rods outer / inner diameter (m)
                 ):
        
        self.dofs_per_node = 3
        self.GD = self.geo_dimension()
        self.mesh = self.init_mesh() 
        
        # Tube section geometry (in meters)
        self.dov, self.div = dov, div
        self.doo, self.dio = doo, dio
        
        self.Av, self.Ao = self.cross_section_area() # Vertical/Other area
        self.Iv, self.Io = self.inertia() # Vertical/Other I
        self.I1, self.I2 = self.structural_inertia()
    
    def __str__(self) -> str:
        """Returns a formatted multi-line string summarizing geometry & mesh information."""
        
        s = f"{self.__class__.__name__}(\n"
        s += "  === Geometry & Mesh ===\n"
        s += f"  Mesh Type             : {self.mesh.__class__.__name__}\n"
        s += f"  Number of Nodes       : {self.mesh.number_of_nodes()}\n"
        s += f"  Number of Elements    : {self.mesh.number_of_cells()}\n"
        s += f"  Geo Dimension         : {self.geo_dimension()}\n"
        s += "  === Section Properties ===\n"
        s += f"  Vertical  rods: A = {self.Av:.6e}, I = {self.Iv:.6e}\n"
        s += f"  Other     rods: A = {self.Ao:.6e}, I = {self.Io:.6e}\n"
        s += "  === Structural Inertia ===\n"
        s += f"  I1 (depth dir) = {self.I1:.6e} m⁴\n"
        s += f"  I2 (width dir) = {self.I2:.6e} m⁴\n"
        s += ")"
        return s
    
    def geo_dimension(self) -> int:
        """Return the geometric dimension of the model."""
        return 3
    
    def cross_section_area(self) -> Tuple[float, float]:
        """Compute the cross-sectional area (A) of tube sections.
            
        Note:
             A = π(do² - di²) / 4
        """
        A_vertical = bm.pi * (self.dov**2 - self.div**2) / 4
        A_other = bm.pi * (self.doo**2 - self.dio**2) / 4
        return A_vertical, A_other
    
    def inertia(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Compute area moment of inertia.
        
        Note: 
            I = π(do⁴ - di⁴) / 64 
        """
        I_vertical = bm.pi * (self.dov**4 - self.div**4) / 64
        I_other = bm.pi * (self.doo**4 - self.dio**4) / 64
        return I_vertical, I_other
    
    def structural_inertia(self) -> Tuple[float, float]:
        """Compute structural area moment of inertia for buckling analysis.
        
        Returns:
            (I1, I2) - Area moments of inertia about different axes
            I1 = 4*A1*(depth/2)**2, where A1 = 4*A_vertical
            I2 = 4*A1*(width/2)**2  where A1 = 4*A_vertical
        """
        A1 = 4*self.Av  # Total area of all 4 vertical columns
        
        # From init_mesh parameters: Wy=0.40 (depth), Wx=0.45 (width)
        depth = 0.40  # Y direction
        width = 0.45  # X direction
        
        I1 = 4 * A1 * (depth/2)**2 
        I2 = 4 * A1 * (width/2)**2 
        
        return I1, I2
    
    def init_mesh(self):
        """Generate a 3D slender truss tower along the z-axis using line (1D) elements.
        
        Parameters:
            n_panel(int):  Number of panels along the z-direction (must be >= 1).
            Lz(float): Total height of the truss structure along the z-axis.
            Wx(float): Width of the rectangular cross-section in the x-direction.
            Wy(float): Depth of the rectangular cross-section in the y-direction.
            lc(float): Characteristic length for mesh size control in Gmsh.
            ne_per_bar(int): Number of divisions per bar element along its length (must be >= 1).
            face_diag(bool): If True, add diagonal bracings on the four side faces.
            
        Return:
            node : (N, 3) ndarray of float. Coordinates of mesh nodes.
            cell : (E, 2) ndarray of int.

        """
        node, cell = TrussTower.build_truss_3d_zbar(n_panel=19, Lz=19, Wx=0.45, Wy=0.40, lc=0.1, ne_per_bar=1)
        mesh = EdgeMesh(node, cell)
        return mesh
    
    def external_load(self, load_total: float=1.0):
        """Generate the external load vector for the 3D truss tower.
        
        Note:
            - Total load = 1 N (load = 1).
            - Applied vertically downward on top nodes.
            - Each top node gets an equal fraction of the total load.
        """
        node = self.mesh.entity('node')
        num_nodes = self.mesh.number_of_nodes()
        
        F = bm.zeros((num_nodes*self.dofs_per_node,), dtype=bm.float64)
        
        # 找到塔顶节点
        top_nodes = bm.where(node[:, 2] > 18.999)[0]
    
        # 每个顶节点施加总载荷的等分
        load_per_node = load_total / len(top_nodes)  # 总载荷: load_total 
        for i in top_nodes:
             F[3*i + 2] = -load_per_node  # Z方向,向下
        
        return F
    
    def dirichlet_dof(self):
        """Return Dirichlet boundary fixed DOFs for the 3D truss tower.
        
        Note:
            All degrees of freedom (ux, uy, uz) at the bottom nodes are fixed.
        """
        node = self.mesh.entity('node')
        
        node_dofs = []
        
        # 找到底部节点（z = 0）
        bottom_nodes = bm.where(node[:, 2] < 1e-6)[0]
        
        # 构造固定自由度：按节点顺序，每个节点依次排列ux, uy, uz
        for node_idx in bottom_nodes:
            dof_x = node_idx * 3       
            dof_y = node_idx * 3 + 1   
            dof_z = node_idx * 3 + 2 
            node_dofs.append(bm.array([dof_x, dof_y, dof_z]))
        
        fixed_dofs = bm.concatenate(node_dofs)
        
        return fixed_dofs
    
    def is_dirichlet_boundary(self):
        """Return boolean array indicating which DOFs are on the boundary.
        
        Returns:
            Array of shape (gdof,) with True for boundary DOFs
        """
        return self.dirichlet_dof()

    def dirichlet_bc(self):
        """Return dirichlet boundary conditions.
        
        Returns:
            Dirichlet values at boundary DOFs (all zeros for fixed boundary)
        """
        fixed_dofs = self.dirichlet_dof()
        return bm.zeros(len(fixed_dofs), dtype=bm.float64)