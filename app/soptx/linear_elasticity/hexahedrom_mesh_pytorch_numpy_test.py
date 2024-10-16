from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from fealpy.decorator import cartesian

from fealpy.mesh import HexahedronMesh, QuadrangleMesh

from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.material.elastic_material import LinearElasticMaterial

from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC

from fealpy.decorator import cartesian

from fealpy.solver import cg

from app.soptx.soptx.utilfs.timer import timer

import argparse

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

parser = argparse.ArgumentParser(description="Solve linear elasticity problems \
                            in arbitrary order Lagrange finite element space on HexahedronMesh.")
parser.add_argument('--backend', 
                    default='pytorch', type=str,
                    help='Specify the backend type for computation, default is "pytorch".')
parser.add_argument('--degree', 
                    default=2, type=int, 
                    help='Degree of the Lagrange finite element space, default is 1.')
parser.add_argument('--nx', 
                    default=2, type=int, 
                    help='Initial number of grid cells in the x direction, default is 2.')
parser.add_argument('--ny',
                    default=2, type=int,
                    help='Initial number of grid cells in the y direction, default is 2.')
parser.add_argument('--nz',
                    default=2, type=int,
                    help='Initial number of grid cells in the z direction, default is 2.')
args = parser.parse_args()

pde = BoxDomainPolyUnloaded3d()
args = parser.parse_args()

bm.set_backend(args.backend)

nx, ny, nz = args.nx, args.ny, args.nz

mesh_cpu = HexahedronMesh.from_box(box=pde.domain(), nx=nx, ny=ny, nz=nz, device='cpu')
mesh_cuda = HexahedronMesh.from_box(box=pde.domain(), nx=nx, ny=ny, nz=nz, device='cuda')

mesh_quad_cpu = QuadrangleMesh.from_box(box=pde.domain(), nx=nx, ny=ny, device='cpu')
mesh_quad_cuda = QuadrangleMesh.from_box(box=pde.domain(), nx=nx, ny=ny, device='cuda')

p = args.degree

tmr = timer("FEM Solver")
next(tmr)

maxit = 4

errorType_cpu = ['$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$']
errorMatrix_cpu = bm.zeros((len(errorType_cpu), maxit), dtype=bm.float64, device=bm.get_device(mesh_cpu))
NDof_cpu = bm.zeros(maxit, dtype=bm.int32, device=bm.get_device(mesh_cpu))
errorType_cuda = ['$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$']
errorMatrix_cuda = bm.zeros((len(errorType_cuda), maxit), dtype=bm.float64, device=bm.get_device(mesh_cuda))
NDof_cuda = bm.zeros(maxit, dtype=bm.int32, device=bm.get_device(mesh_cuda))

for i in range(maxit):
    space_cpu = LagrangeFESpace(mesh_cpu, p=p, ctype='C')
    tensor_space_cpu = TensorFunctionSpace(space_cpu, shape=(-1, 3))
    NDof_cpu[i] = tensor_space_cpu.number_of_global_dofs()
    tmr.send('space_cpu')
    linear_elastic_material_cpu = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='3D')
    tmr.send('material_cpu')
    space_cuda = LagrangeFESpace(mesh_cuda, p=p, ctype='C')
    tensor_space_cuda = TensorFunctionSpace(space_cuda, shape=(-1, 3))
    NDof_cuda[i] = tensor_space_cuda.number_of_global_dofs()
    tmr.send('space_cuda')
    linear_elastic_material_cuda = LinearElasticMaterial(name='lam1_mu1',
                                                lame_lambda=1, shear_modulus=1,
                                                hypo='3D')
    tmr.send('material_cuda')

    integrator_K_cpu = LinearElasticIntegrator(material=linear_elastic_material_cpu, 
                                            q=tensor_space_cpu.p+3)
    bform_cpu = BilinearForm(tensor_space_cpu)
    bform_cpu.add_integrator(integrator_K_cpu)
    K_cpu = bform_cpu.assembly(format='csr')
    tmr.send('stiffness assembly_cpu')

    integrator_K_cuda = LinearElasticIntegrator(material=linear_elastic_material_cuda,
                                            q=tensor_space_cuda.p+3)
    bform_cuda = BilinearForm(tensor_space_cuda)
    bform_cuda.add_integrator(integrator_K_cuda)
    K_cuda = bform_cuda.assembly(format='csr')
    tmr.send('stiffness assembly_cuda')

    integrator_F_cpu = VectorSourceIntegrator(source=pde.source, q=tensor_space_cpu.p+3)
    lform_cpu = LinearForm(tensor_space_cpu)    
    lform_cpu.add_integrator(integrator_F_cpu)
    F_cpu = lform_cpu.assembly()
    tmr.send('source assembly_cpu')

    integrator_F_cuda = VectorSourceIntegrator(source=pde.source, q=tensor_space_cuda.p+3)
    lform_cuda = LinearForm(tensor_space_cuda)
    lform_cuda.add_integrator(integrator_F_cuda)
    F_cuda = lform_cuda.assembly()
    # error_F1 = bm.sum(bm.abs(F_cpu - bm.device_put(F_cuda, device='cpu')))
    # error_K1 = bm.sum(bm.abs(K_cpu.to_dense() - bm.device_put(K_cuda.to_dense(), device='cpu')))
    tmr.send('source assembly_cuda')

    uh_bd_cpu = bm.zeros(tensor_space_cpu.number_of_global_dofs(), 
                        dtype=bm.float64, device=bm.get_device(mesh_cpu))
    uh_bd_cpu, isDDof_cpu = tensor_space_cpu.boundary_interpolate(gD=pde.dirichlet, 
                                                                uh=uh_bd_cpu, threshold=None)
    F_cpu = F_cpu - K_cpu.matmul(uh_bd_cpu)
    F_cpu[isDDof_cpu] = uh_bd_cpu[isDDof_cpu]
    dbc_cpu = DirichletBC(space=tensor_space_cpu)
    K_cpu = dbc_cpu.apply_matrix(matrix=K_cpu, check=True)
    tmr.send('boundary_cpu')

    uh_bd_cuda = bm.zeros(tensor_space_cuda.number_of_global_dofs(), 
                        dtype=bm.float64, device=bm.get_device(mesh_cuda))
    uh_bd_cuda, isDDof_cuda = tensor_space_cuda.boundary_interpolate(gD=pde.dirichlet, 
                                                                    uh=uh_bd_cuda, threshold=None)
    F_cuda = F_cuda - K_cuda.matmul(uh_bd_cuda)
    F_cuda[isDDof_cuda] = uh_bd_cuda[isDDof_cuda]
    dbc_cuda = DirichletBC(space=tensor_space_cuda)
    K_cuda = dbc_cuda.apply_matrix(matrix=K_cuda, check=True)

    # error_isDDof = bm.where(isDDof_cpu != bm.device_put(uh_bd_cuda, device='cpu'))
    # error_uh_bd = bm.sum(bm.abs(uh_bd_cpu - bm.device_put(uh_bd_cuda, device='cpu')))
    # error_F2 = bm.sum(bm.abs(F_cpu - bm.device_put(F_cuda, device='cpu')))
    # error_K2 = bm.sum(bm.abs(K_cpu.to_dense() - bm.device_put(K_cuda.to_dense(), device='cpu')))
    tmr.send('boundary_cuda')  

    uh_cpu = tensor_space_cpu.function()
    uh_cpu[:] = cg(K_cpu, F_cpu, maxiter=1000, atol=1e-14, rtol=1e-14)
    tmr.send('solve(cg)_cpu')

    uh_cuda = tensor_space_cuda.function()
    uh_cuda[:] = cg(K_cuda, F_cuda, maxiter=1000, atol=1e-14, rtol=1e-14)
    tmr.send('solve(cg)_cuda')

    u_exact_cpu = tensor_space_cpu.interpolate(pde.solution)
    errorMatrix_cpu[0, i] = bm.sqrt(bm.sum(bm.abs(uh_cpu[:] - u_exact_cpu)**2 * (1 / NDof_cpu[i])))
    errorMatrix_cpu[1, i] = mesh_cpu.error(u=uh_cpu, v=pde.solution, q=tensor_space_cpu.p+3, power=2)
    tmr.send('error_cpu')

    u_exact_cuda = tensor_space_cuda.interpolate(pde.solution)
    errorMatrix_cuda[0, i] = bm.sqrt(bm.sum(bm.abs(uh_cuda[:] - u_exact_cuda)**2 * (1 / NDof_cuda[i])))
    errorMatrix_cuda[1, i] = mesh_cuda.error(u=uh_cuda, v=pde.solution, q=tensor_space_cuda.p+3, power=2)
    tmr.send('error_cuda')

    error_uh = bm.sum(bm.abs(uh_cpu[:] - bm.device_put(uh_cuda[:], device='cpu')))
    error_exact = bm.sum(bm.abs(u_exact_cpu - bm.device_put(u_exact_cuda, device='cpu')))
    
    tmr.send(None)

    if i < maxit-1:
        mesh_cpu.uniform_refine()
        mesh_cuda.uniform_refine()

print("errorMatrix_cpu:\n", errorType_cpu, "\n", errorMatrix_cpu)
print("errorMatrix_cuda:\n", errorType_cuda, "\n", errorMatrix_cuda)
print("order_l2_cpu:\n", bm.log2(errorMatrix_cpu[0, :-1] / errorMatrix_cpu[0, 1:]))
print("order_l2_cuda:\n", bm.log2(errorMatrix_cuda[0, :-1] / errorMatrix_cuda[0, 1:]))
print("order_L2_cpu:\n ", bm.log2(errorMatrix_cpu[1, :-1] / errorMatrix_cpu[1, 1:]))
print("order_L2_cuda:\n ", bm.log2(errorMatrix_cuda[1, :-1] / errorMatrix_cuda[1, 1:]))