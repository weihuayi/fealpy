from typing import Sequence
from ...decorator import cartesian,variantmethod
from ...backend import backend_manager as bm
from ...backend import TensorLike

class CircleInterfaceData2D:
    """
    2D Allen-Cahn phase field:

        -\phi_t + (u \cdot \nabla)\phi = \gamma(\Delta \phi - f(\phi))          (x, y) \in \Omega, t > 0
        \phi(x, y, 0) =   -tanh((\sqrt{x^2 + y^2} - r_0)/\eta)                  (x, y) \in \Omega
        where \Omega = \{(x, y) | x^2 + y^2 < r_0^2\} is a circle of radius r_0 centered at the origin.
    
    Exact solution:
        Have no exact solution, but we can use the initial condition as a reference.

    """
    def __init__(self,u = 0):
        self.box = [-1, 1, -1, 1]
        self.r0 = 100/128
        self.gam = 6.10351e-05
        self.eta = 0.0078
        self.area = 4

    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return self.box  

    @variantmethod('tri')
    def init_mesh(self, **kwargs):
        """
        Initialize the mesh with given number of points in x and y directions.
        """
        from ...mesh import TriangleMesh
        nx = 256
        ny = 256
        mesh = TriangleMesh.from_box(self.box, nx=nx, ny=ny, **kwargs)
        return mesh
    
    @init_mesh.register('quad')
    def init_mesh(self, **kwargs):
        from ...mesh import QuadrangleMesh
        nx = 256
        ny = 256
        mesh = QuadrangleMesh.from_box(self.box, nx=nx, ny=ny, **kwargs)
        return mesh
    
    @init_mesh.register('moving_tri')
    def init_mesh(self, **kwargs):
        """
        Initialize the mesh with given number of points in x and y directions.
        """
        from ...mesh import TriangleMesh
        nx = 64
        ny = 64
        mesh = TriangleMesh.from_box(self.box, nx=nx, ny=ny, **kwargs)

        domain = self.box
        vertices = bm.array([[domain[0], domain[2]],
                             [domain[1], domain[2]],
                             [domain[1], domain[3]],
                             [domain[0], domain[3]]], **kwargs)
        mesh.nodedata['vertices'] = vertices
        return mesh
    
    @init_mesh.register('moving_quad')
    def init_mesh(self, **kwargs):
        from ...mesh import QuadrangleMesh
        nx = 64
        ny = 64
        mesh = QuadrangleMesh.from_box(self.box, nx=nx, ny=ny, **kwargs)

        domain = self.box
        vertices = bm.array([[domain[0], domain[2]],
                             [domain[1], domain[2]],
                             [domain[1], domain[3]],
                             [domain[0], domain[3]]], **kwargs)
        mesh.nodedata['vertices'] = vertices
        return mesh
    
    @init_mesh.register('moving_tri_unstru')
    def init_mesh(self, **kwargs):
        domain = self.box
        vertices = bm.array([[domain[0], domain[2]],
                             [domain[1], domain[2]],
                             [domain[1], domain[3]],
                             [domain[0], domain[3]]], **kwargs)

        from ...mesh import TriangleMesh
        h = 0.04
        mesh = TriangleMesh.from_polygon_gmsh(vertices=vertices,h=h, **kwargs)
        mesh.nodedata['vertices'] = vertices
        return mesh
    
    def duration(self) -> Sequence[float]:
        """the time interval [t0, t1]."""
        return [0.0, 5000.0]
    
    def nonlinear_source(self, phi):
        """Return the nonlinear source term f(φ) = 1/η^2 *(φ^3 - φ)."""
        eta = self.eta
        return (phi**3 - phi) / (eta**2)
    
    def gamma(self) -> float:
        """Return the gamma parameter in the Allen-Cahn equation."""
        return self.gam
    
    def velocity_field(self, p: TensorLike,t = 0.0) -> TensorLike:
        """Return the velocity field u."""
        x = p[..., 0]
        y = p[..., 1]
        return bm.zeros_like(p, dtype=bm.float64)

    @cartesian
    def init_condition(self, p: TensorLike,t = 0.0) -> TensorLike:
        x = p[..., 0]
        y = p[..., 1]
        r = bm.sqrt((x)**2 + (y)**2)
        r0 = self.r0
        eta = self.eta
        return -bm.tanh((r - r0) / eta)
    