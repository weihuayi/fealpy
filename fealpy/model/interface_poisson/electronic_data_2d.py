from typing import Sequence
from ...decorator import cartesian, variantmethod
from ...backend import TensorLike
from ...backend import backend_manager as bm

class ElectronicData2D:
    """
    2D elliptic interface problem:

        (∂²u/∂x² + ∂²u/∂y²) = f   (x, y) ∈ [1, 10] x [-10, 10], 

    Level function:

        Φ(x, y) = x² + y² - (2.1)²

    Exact solution:

        u^+(x, y) = -x + (2.1²)x/(x² + y²)  ∀x, y ∈ Ω+
        u^-(x, y) = 0                       ∀x, y ∈ Ω-

    The corresponding source term is:    

        f^+(x, y) = 0
        f^-(x, y) = 0
    
    Interface condition:
        [u]_Γ = u^+ - u^- = q. [u_n]_Γ = ▽u^+_n - ▽u^-_n
    Homogeneous Dirichlet boundary conditions are applied on all edges.
    """
    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 2

    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax]."""
        return [1, 10, -10, 10]  
    
    def level_function(self, p: TensorLike) -> TensorLike:
        """Check if point is in Ω+ or Ω-."""
        x = p[...,0]
        y = p[...,1]

        return x**2 + y**2 - 2.1**2

    @variantmethod('uniform_tri')
    def init_mesh(self, nx=10, ny=10):
        domain = self.domain()
        hx = (domain[1] - domain[0])/nx
        hy = (domain[3] - domain[2])/ny
        from ...mesh import UniformMesh, TriangleMesh
        if hasattr(self, 'level_function'):
            back_mesh = UniformMesh(extent=(0, nx, 0, ny), h=(hx, hy), origin=(domain[0], domain[2]))
            mesh = TriangleMesh.interfacemesh_generator(back_mesh, self.level_function)
            return mesh, back_mesh
        else:
            mesh = TriangleMesh.from_box(box=self.domain, nx=nx, ny=ny)
            return mesh
            
    @cartesian
    def solution(self, p: TensorLike) -> TensorLike:
        """Compute exact solution. """
        kwargs = bm.context(p)
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi

        val = bm.zeros(x.shape, **kwargs)
        Omega0 = self.level_function(p)>0
        Omega1 = self.level_function(p)<0
        
        val[Omega0] = -x[Omega0] + 2.1**2*x[Omega0]/(x[Omega0]**2 + y[Omega0]**2)
        val[Omega1] = 0

        return val

    @cartesian
    def gradient(self, p: TensorLike) -> TensorLike:
        """Compute gradient of solution."""
        kwargs = bm.context(p)
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi

        grad0 = bm.zeros(x.shape, **kwargs)
        grad1 = bm.zeros(x.shape, **kwargs)

        Omega0 = self.level_function(p)>0 
        Omega1 = self.level_function(p)<0 

        grad0[Omega0] = 4.41*(-x[Omega0]**2 + y[Omega0]**2)/(x[Omega0]**2 + y[Omega0]**2)**2 - 1
        grad0[Omega1] = 0

        grad1[Omega0] = -2*2.1**2*(y[Omega0]*x[Omega0])/(x[Omega0]**2 + y[Omega0]**2)**2
        grad1[Omega1] = 0

        return bm.stack((grad0, grad1), axis=-1)

    @cartesian
    def gN(self, p: TensorLike)-> TensorLike:
        """Interface flux jump conditon gN"""
        kwargs = bm.context(p)
        x, y = p[..., 0], p[..., 1]

        n2 = bm.zeros(x.shape, **kwargs)
        n2 = (2*x)/2.1

        return n2

    @cartesian
    def source(self, p: TensorLike) -> TensorLike:
        """Compute exact source"""
        kwargs = bm.context(p)
        x = p[..., 0]
        y = p[..., 1]
        pi = bm.pi

        val = bm.zeros(x.shape, **kwargs)
                
        return val 
    
    @cartesian
    def dirichlet(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p: TensorLike) -> TensorLike:
        """Check if point is on boundary."""
        x, y = p[..., 0], p[..., 1]
        return (bm.abs(x - 1.0) < 1e-12) | (bm.abs(x - 10.0) < 1e-12) | \
               (bm.abs(y + 10.0) < 1e-12) | (bm.abs(y - 10.0) < 1e-12)
