from fealpy.typing import Tuple, TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesh import EdgeMesh
from ...mesh import TrussTower


class TrussTowerData3D:
    """
    """
    def __init__(self):
        
        self.dofs_per_node = 3
        self.mesh = self.init_mesh() 
    
    def __str__(self) -> str:
        """Returns a formatted multi-line string summarizing the configuration of the 3D truss tower data.
        
        Returns:
            str: A multi-line string showing key truss parameters, geometry, and mesh info.
    """
        s = f"{self.__class__.__name__}(\n"
        s += "  === Geometry & Mesh ===\n"
        s += f"  Mesh Type             : {self.mesh.__class__.__name__}\n"
        s += f"  Number of Nodes       : {self.mesh.number_of_nodes()}\n"
        s += f"  Number of Elements    : {self.mesh.number_of_cells()}\n"
        s += f"  Geo Dimension         : {self.geo_dimension()}\n"
        s += ")"
        return s
    
    def geo_dimension(self) -> int:
        """the geometric dimension."""
        return 3
    
    def inertia(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """moments of inertia."""
        pass 
    
    
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
    
    def load(self):
        pass
    
    def dirichlet_dof(self):
        pass