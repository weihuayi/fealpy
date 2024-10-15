from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from fealpy.decorator import cartesian

from fealpy.mesh import HexahedronMesh, QuadrangleMesh

from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

class BoxDomainPolyUnloaded3d():
    def __init__(self):
        pass
        
    def domain(self):
        return [0, 1, 0, 1, 0, 1]
    
    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=points.device)
        val[..., 0] = 2*x**3 - 3*x*y**2 - 3*x*z**2
        val[..., 1] = 2*y**3 - 3*y*x**2 - 3*y*z**2
        val[..., 2] = 2*z**3 - 3*z*y**2 - 3*z*x**2
        
        return val

    @cartesian
    def source(self, points: TensorLike):
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=points.device)
        
        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return self.solution(points)
    
class BoxDomainPolyLoaded3d():
    def domain(self):
        return [0, 1, 0, 1, 0, 1]
    
    @cartesian
    def source(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=points.device)
        mu = 1
        factor1 = -400 * mu * (2 * y - 1) * (2 * z - 1)
        term1 = 3 * (x ** 2 - x) ** 2 * (y ** 2 - y + z ** 2 - z)
        term2 = (1 - 6 * x + 6 * x ** 2) * (y ** 2 - y) * (z ** 2 - z)
        val[..., 0] = factor1 * (term1 + term2)

        factor2 = 200 * mu * (2 * x - 1) * (2 * z - 1)
        term1 = 3 * (y ** 2 - y) ** 2 * (x ** 2 - x + z ** 2 - z)
        term2 = (1 - 6 * y + 6 * y ** 2) * (x ** 2 - x) * (z ** 2 - z)
        val[..., 1] = factor2 * (term1 + term2)

        factor3 = 200 * mu * (2 * x - 1) * (2 * y - 1)
        term1 = 3 * (z ** 2 - z) ** 2 * (x ** 2 - x + y ** 2 - y)
        term2 = (1 - 6 * z + 6 * z ** 2) * (x ** 2 - x) * (y ** 2 - y)
        val[..., 2] = factor3 * (term1 + term2)

        return val

    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=points.device)

        mu = 1
        val[..., 0] = 200*mu*(x-x**2)**2 * (2*y**3-3*y**2+y) * (2*z**3-3*z**2+z)
        val[..., 1] = -100*mu*(y-y**2)**2 * (2*x**3-3*x**2+x) * (2*z**3-3*z**2+z)
        val[..., 2] = -100*mu*(z-z**2)**2 * (2*y**3-3*y**2+y) * (2*x**3-3*x**2+x)

        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, 
                        dtype=points.dtype, device=points.device)

pde = BoxDomainPolyUnloaded3d()

bm.set_backend('pytorch')

p = 1
nx, ny, nz = 2, 2, 2

mesh_hex_cpu = HexahedronMesh.from_box(box=pde.domain(), nx=nx, ny=ny, nz=nz, device='cpu')
mesh_hex_cuda = HexahedronMesh.from_box(box=pde.domain(), nx=nx, ny=ny, nz=nz, device='cuda')

mesh_quad_cpu = QuadrangleMesh.from_box(box=pde.domain(), nx=nx, ny=ny, device='cpu')
mesh_quad_cuda = QuadrangleMesh.from_box(box=pde.domain(), nx=nx, ny=ny, device='cuda')

space_hex_cpu = LagrangeFESpace(mesh_hex_cpu, p=p, ctype='C')
tensor_space_hex_cpu = TensorFunctionSpace(space_hex_cpu, shape=(-1, 3))

edge_quad_cpu = mesh_quad_cpu.edge
edge_quad_cuda = mesh_quad_cuda.edge

edge_hex_cpu = mesh_hex_cpu.edge
edge_hex_cuda = mesh_hex_cuda.edge


edge2ipoint_hex_cpu = mesh_hex_cpu.edge_to_ipoint(p=p)
edge2ipoint_quad_cpu = mesh_quad_cpu.edge_to_ipoint(p=p)

print("---------------")