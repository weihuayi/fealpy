from typing import Sequence, Optional
from ...backend import backend_manager as bm
from ...decorator import cartesian, variantmethod
from ...typing import  TensorLike
from ...mesh import HexahedronMesh

class BoxPolyData3d():
    """
    3D Linear Elasticity problem

    -∇·σ = f  in Ω
    where σ = λ(∇·u)I + 2με is the stress tensor, ε is the strain tensor.

    with the displacement field is:
    u_x = 1e-3 * (2x + y + z) / 2
    u_y = 1e-3 * (x + 2y + z) / 2
    u_z = 1e-3 * (x + y + 2z) / 2

    Material properties:
    - λ = 4e5 MPa (First Lamé parameter)
    - μ = 4e5 MPa (Shear modulus)

    Boundary conditions:
    - Non-homogeneous Dirichlet boundary conditions on all boundaries ∂Ω
    - u = u_exact on ∂Ω (prescribed displacement from the exact solution)
    - No traction (Neumann) boundary conditions
    """
    def geo_dimension(self) -> int:
        """Return the geometric dimension of the domain."""
        return 3
    
    def domain(self) -> Sequence[float]:
        """Return the computational domain [xmin, xmax, ymin, ymax, zmin, zmax]."""
        return [0, 1, 0, 1, 0, 1]

    @variantmethod('hex')
    def init_mesh(self):
        node = bm.array([[0.249, 0.342, 0.192],
                        [0.826, 0.288, 0.288],
                        [0.850, 0.649, 0.263],
                        [0.273, 0.750, 0.230],
                        [0.320, 0.186, 0.643],
                        [0.677, 0.305, 0.683],
                        [0.788, 0.693, 0.644],
                        [0.165, 0.745, 0.702],
                        [0, 0, 0],
                        [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [1, 0, 1],
                        [1, 1, 1],
                        [0, 1, 1]],
                    dtype=bm.float64)

        cell = bm.array([[0, 1, 2, 3, 4, 5, 6, 7],
                        [0, 3, 2, 1, 8, 11, 10, 9],
                        [4, 5, 6, 7, 12, 13, 14, 15],
                        [3, 7, 6, 2, 11, 15, 14, 10],
                        [0, 1, 5, 4, 8, 9, 13, 12],
                        [1, 2, 6, 5, 9, 10, 14, 13],
                        [0, 4, 7, 3, 8, 12, 15, 11]],
                        dtype=bm.int32)
        mesh = HexahedronMesh(node, cell)

        return mesh

    @init_mesh.register('tet')
    def init_mesh(self):
        pass

    def lam(self, p: Optional[TensorLike] = None) -> TensorLike:
        """First Lamé parameter λ = 4e5 MPa."""
        return 4e5

    def mu(self, p: Optional[TensorLike] = None) -> TensorLike:
        """Second Lamé parameter μ (shear modulus) = 4e5 MPa."""
        return 4e5

    def rho(self, p: Optional[TensorLike] = None) -> TensorLike:
        pass

    @cartesian
    def body_force(self, p: TensorLike):
        val = bm.zeros(p.shape, dtype=p.dtype, device=bm.get_device(p))
        return val

    @cartesian
    def displacement(self, p: TensorLike):
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        u_x = 1e-3 * (2*x + y + z) / 2
        u_y = 1e-3 * (x + 2*y + z) / 2
        u_z = 1e-3 * (x + y + 2*z) / 2
        
        return bm.stack([u_x, u_y, u_z], axis=-1)
    
        val = bm.zeros(p.shape, dtype=p.dtype, device=bm.get_device(p))
    
        val = bm.set_at(val, (..., 0), 1e-3 * (2*x + y + z) / 2)
        val = bm.set_at(val, (..., 1), 1e-3 * (x + 2*y + z) / 2)
        val = bm.set_at(val, (..., 2), 1e-3 * (x + y + 2*z) / 2)

        return val

    @cartesian
    def strain(self, p: TensorLike) -> TensorLike:
        shape = p.shape[:-1] + (3, 3)
        val = bm.zeros(shape, dtype=p.dtype, device=bm.get_device(p))

        val = bm.set_at(val, (..., 0, 0), 1e-3)  
        val = bm.set_at(val, (..., 1, 1), 1e-3)  
        val = bm.set_at(val, (..., 2, 2), 1e-3)  

        # Off-diagonal components (symmetric)
        val = bm.set_at(val, (..., 0, 1), 1e-3 / 2)  
        val = bm.set_at(val, (..., 1, 0), 1e-3 / 2)  
        val = bm.set_at(val, (..., 0, 2), 1e-3 / 2)  
        val = bm.set_at(val, (..., 2, 0), 1e-3 / 2)  
        val = bm.set_at(val, (..., 1, 2), 1e-3 / 2)  
        val = bm.set_at(val, (..., 2, 1), 1e-3 / 2)  

        return val

    @cartesian
    def stress(self, p: TensorLike) -> TensorLike:
        shape = p.shape[:-1] + (3, 3)
        val = bm.zeros(shape, dtype=p.dtype, device=bm.get_device(p))
        
        lam = self.lam(p)
        mu = self.mu(p) 
        
        tr_eps = 3e-3

        val = bm.set_at(val, (..., 0, 0), lam * tr_eps + 2 * mu * 1e-3)  
        val = bm.set_at(val, (..., 1, 1), lam * tr_eps + 2 * mu * 1e-3)  
        val = bm.set_at(val, (..., 2, 2), lam * tr_eps + 2 * mu * 1e-3)  

        val = bm.set_at(val, (..., 0, 1), 2 * mu * 1e-3 / 2) 
        val = bm.set_at(val, (..., 1, 0), 2 * mu * 1e-3 / 2)  
        val = bm.set_at(val, (..., 0, 2), 2 * mu * 1e-3 / 2)  
        val = bm.set_at(val, (..., 2, 0), 2 * mu * 1e-3 / 2)  
        val = bm.set_at(val, (..., 1, 2), 2 * mu * 1e-3 / 2)  
        val = bm.set_at(val, (..., 2, 1), 2 * mu * 1e-3 / 2) 
        
        return val

    @cartesian
    def displacement_bc(self, points: TensorLike) -> TensorLike:
        result = self.displacement(points)

        return result
    
    @cartesian
    def is_displacement_boundary(self, p: TensorLike) -> TensorLike:
        eps = 1e-12
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        
        on_x0 = bm.abs(x - 0.0) < eps
        on_x1 = bm.abs(x - 1.0) < eps
        on_y0 = bm.abs(y - 0.0) < eps
        on_y1 = bm.abs(y - 1.0) < eps
        on_z0 = bm.abs(z - 0.0) < eps
        on_z1 = bm.abs(z - 1.0) < eps
        
        return on_x0 | on_x1 | on_y0 | on_y1 | on_z0 | on_z1
    
    @cartesian
    def traction_bc(self, p: TensorLike) -> TensorLike:
        pass

    @cartesian
    def is_traction_boundary(self, p: TensorLike) -> TensorLike:
        pass
