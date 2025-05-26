from fealpy.mesh import EdgeMesh
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian

class BeamData2D:
    """
    1D Euler-Bernoulli beam problem:

        E I d⁴u/dx⁴ = f,    x ∈ (0, L)
        u(0) = u(L) = 0,    (clamped or simply supported ends)
        u''(0) = u''(L) = 0 (for simply supported ends)
        f(x) = constant distributed load

    Parameters:
        E: Young's modulus
        I: Moment of inertia
        A: Cross-sectional area (used if axial terms are present)
        f: Distributed load (constant)
        L: Beam length
        n: Number of mesh elements

    This class constructs a 1D mesh and provides the source term.
    """

    def __init__(self, E=200e9, I=118.6e-6, A=10.3, f=-25000, L=10):
        """
        Initialize beam parameters.

        Args:
            E (float): Young's modulus.
            I (float): Moment of inertia.
            A (float): Cross-sectional area.
            f (float): Distributed load.
            L (float): Beam length.
            n (int): Number of mesh elements.
        """
        self.E = E
        self.I = I
        self.A = A
        self.f = f
        self.L = L

        self.mesh = self.init_mesh()

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

        Args:
            x: Spatial coordinate(s).

        Returns:
            Tensor: Distributed load at x.
        """
        return bm.ones_like(x) * self.f
    
    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief Dirichlet 边界的判断函数
        固定左端：x = 0 处固定
        参数:
            p: ndarray, shape (..., geo_dim)
               坐标点数组
        返回:
            is_dirichlet: ndarray[bool]
        """
        x = p[..., 0]
        return bm.abs(x - 0.0) < 1e-12
    
    @cartesian
    def dirichlet(self, x):
        """
        Compute the Dirichlet boundary condition.

        Args:
            x: Spatial coordinate(s).

        Returns:
            Tensor: Dirichlet boundary condition at x.
        """
        return bm.zeros_like(x)
