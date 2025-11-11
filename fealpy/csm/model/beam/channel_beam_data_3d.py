from fealpy.typing import Tuple, TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.mesh import EdgeMesh


class ChannelBeamData3D:
    """
    """
    def __init__(self, mu_y: float=2.44, mu_z: float=2.38):
        
        self.FSY = mu_y
        self.FSZ = mu_z
            
        self.A = self.cross_section()
        self.Ix, self.Iy, self.Iz = self.inertia()
        
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
        s += f"  Geo Dimension         : {self.geo_dimension()}\n"
        s += f"  Shear Factors     : {self.FSY}, {self.FSZ}\n"
        s += f"  A : {self.A} ...\n"
        s += f"  Ix, Iy, Iz : {self.Ix[:5].tolist()} ...\n"
        s += ")"
        return s
    
    def geo_dimension(self) -> int:
        """the geometric dimension."""
        return 3
    
    def cross_section(self) -> float:
        """Beam cross-sectional areas."""
        return  4.90e-4
    
    def inertia(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Beam moments of inertia."""
        Ix = 5.18e-9
        Iy = 2.77e-8
        Iz = 1.69e-7
        return Ix, Iy, Iz
    
    def init_mesh(self):
        return 
    
    
    def load(self):
        pass
    
    def dirichlet_dof(self):
        pass
     
    
    