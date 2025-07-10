from fealpy.mesh import EdgeMesh
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike

class BarData1D:
    """
    1D Axial Bar Problem:

        E A du/dx = p(x),    x ∈ (0, L)
        u(0) = 0,            (fixed boundary at node 4)
        p(x) = constant axial force

    Parameters:
        E: Young's modulus
        A: Cross-sectional area (used if axial terms are present)
        f: load 
        L: Bar length
        n: Number of mesh elements

    This class constructs a 1D axial bar mesh with specified boundary conditions and external loads.
    It also provides the source term for the axial force distribution.
    """

    def __init__(self):
        """
        Initialize bar parameters.

        Parameters:
            E (float): Young's modulus.
            A (float): Cross-sectional area.
            f (float): load.
            L (float): Beam length.
            n (int): Number of mesh elements.
        """
        self.mesh = self.init_mesh()

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 1

    def domain(self):
        """Return the computational domain [xmin, xmax]."""
        return [0.0, self.L]

    def init_mesh(self):
        """
        Construct a 1D EdgeMesh for the bar domain.

        Returns:
            EdgeMesh: 1D mesh from x=0 to x=L.
        """
        node = bm.array([[0], [0.1],[0.2], [0.3]], dtype=bm.float64)
        cell = bm.array([[0, 1],[1,2], [2, 3]] , dtype=bm.int32)
        return EdgeMesh(node, cell)
    
    @cartesian
    def source(self, x):
        """
        Compute the load vector f(x) based on the external forces applied at specific nodes.

        Args:
            x: Spatial coordinate(s).

        Returns:
            Tensor: Load vector f at x.
        """
        # Initialize the load vector as zero
        f = bm.zeros_like(x)
        
        # Apply point loads at the corresponding nodes
        # Apply f1 at node 1 (x=0)
        f += bm.where(x == 0, self.f[0], 0)  

        # Apply f2 at node 2 (x=l1)
        f += bm.where(x == self.init_mesh[1], self.ff[1], 0)  

        # Apply p3 at node 3 (x=l1 + l2)
        f += bm.where(x == self.init_mesh[1] + self.init_mesh[2], self.f[2], 0)  

        return f
    
    def dirichlet_dof_index(self) -> TensorLike:
        """
       The indices of degrees of freedom (DOFs) where Dirichlet boundary conditions are applied.
        
        Parameters:
            total_dof : int
            Total number of global degrees of freedom.

        Returns:
            Tensor[int]: Indices of boundary DOFs.
        """
        return bm.array([0, 1, 2, 3])  
    
    @cartesian
    def dirichlet(self, x):
        """
        Compute the Dirichlet boundary condition.

        Args:
            x: Spatial coordinate(s).

        Returns:
            Tensor: Dirichlet boundary condition at x.
        """
        return bm.zeros((x.shape[0], 3))
