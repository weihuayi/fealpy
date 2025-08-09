from fealpy.mesh import EdgeMesh
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.backend import TensorLike

class EulerBernoulliBeamData2D:
    """
    1D Euler-Bernoulli beam problem

        E I d⁴u/dx⁴ = f,    x ∈ (0, L)
        u(0) = u(L) = 0,    (clamped or simply supported ends)
        u''(0) = u''(L) = 0 (for simply supported ends)
        f(x) = constant distributed load

    Parameters
        E: Young's modulus
        I: Moment of inertia
        A: Cross-sectional area (used if axial terms are present)
        f: Distributed load (constant)
        L: Beam length
        n: Number of mesh elements

    This class constructs a 1D mesh and provides the source term.
    """

    def __init__(self):
        """
        Initialize beam parameters.

        Parameters
            E (float): Young's modulus.
            I (float): Moment of inertia.
            A (float): Cross-sectional area.
            f (float): Distributed load.
            L (float): Beam length.
            n (int): Number of mesh elements.
        """
        self.mesh = self.init_mesh()
        self.E = 200e9  # Young's modulus in Pascals
        self.I = 118.6e-6  # Moment of inertia in m^
        self.A = 10.3  # Cross-sectional area in m^2
        self.f = -25000  # Distributed load in N/m
        self.L = 10.0  # Beam length in meters
    
    def __str__(self) -> str:
        """Return a multi-line summary including PDE type and key params."""
        return (
            f"\n  euler_bernoulli (2D Euler-Bernoulli PDE on  domain)\n"
            f"  Box dimensions: L = {self.L}\n"
            f"  young's modulus: E = {self.E}\n"
            f"  moment of inertia: I = {self.I}\n"
            f"  cross-sectional area: A = {self.A}\n"
            f"  distributed load: f = {self.f}\n"
        )

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 1

    def domain(self):
        """Return the computational domain [xmin, xmax]."""
        return [0.0, self.L]

    def init_mesh(self):
        """
        Construct a 1D EdgeMesh for the beam domain.

        Returns:
            EdgeMesh: 1D mesh from x=0 to x=L.
        """
        node = bm.array([[0], [5],[7.5]], dtype=bm.float64)
        cell = bm.array([[0, 1],[1,2]] , dtype=bm.int32)
        return EdgeMesh(node, cell)
    
    @cartesian
    def source(self, x):
        """
        Compute the distributed load f(x).

        Parameters
            x: Spatial coordinate(s).

        Returns
            Tensor: Distributed load at x.
        """
        return bm.ones_like(x) * self.f
    
    
    def dirichlet_dof_index(self) -> TensorLike:
        """
        Return the indices of degrees of freedom (DOFs) where Dirichlet boundary conditions are applied.
        For example, fix all DOFs (such as the first two) at the first node.

        Parameters
            total_dof : int
            Total number of global degrees of freedom.

        Returns
            Tensor[int]: Indices of boundary DOFs.
        """
        return bm.array([0, 1, 2])  
    
    @cartesian
    def dirichlet(self, x):
        """
        Compute the Dirichlet boundary condition.

        Parameters
            x: Spatial coordinate(s).

        Returns
            Tensor: Dirichlet boundary condition at x.
        """
        return bm.zeros((x.shape[0], 2))

