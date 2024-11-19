from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.fem.nonlinear_elastic_integrator import NonlinearElasticIntegrator
from fealpy.decorator import cartesian
from fealpy.solver import cg, spsolve

from app.soptx.soptx.utils.timer import timer

from fealpy.material.elastic_material import LinearElasticMaterial


import argparse

# 平面应变问题
class BoxDomainPolyData2D():

    def domain(self):
        return [0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike, index=None) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[..., 0] = 35/13 * y - 35/13 * y**2 + 10/13 * x - 10/13 * x**2
        val[..., 1] = -25/26 * (-1 + 2 * y) * (-1 + 2 * x)
        
        return val
    
    @cartesian
    def solution(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[..., 0] = x * (1 - x) * y * (1 - y)
        val[..., 1] = 0
        
        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return self.solution(points)

class BoxDomainTriData2D():

    def domain(self):
        return [0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike, index=None) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[..., 0] = (22.5 * bm.pi**2) / 13 * bm.sin(bm.pi * x) * bm.sin(bm.pi * y)
        val[..., 1] = - (12.5 * bm.pi**2) / 13 * bm.cos(bm.pi * x) * bm.cos(bm.pi * y)
        
        return val
    
    @cartesian
    def solution(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[..., 0] = bm.sin(bm.pi * x) * bm.sin(bm.pi * y)
        val[..., 1] = 0
        
        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:
        return self.solution(points)

    

parser = argparse.ArgumentParser(description="Solve linear elasticity problems in arbitrary order Lagrange finite element space on QuadrangleMesh.")
parser.add_argument('--backend',
                    choices=('numpy', 'pytorch'), 
                    default='numpy', type=str,
                    help='Specify the backend type for computation, default is pytorch.')
parser.add_argument('--solver',
                    choices=('cg', 'spsolve'),
                    default='cg', type=str,
                    help='Specify the solver type for solving the linear system, default is "cg".')
parser.add_argument('--degree', 
                    default=2, type=int, 
                    help='Degree of the Lagrange finite element space, default is 2.')
parser.add_argument('--nx', 
                    default=16, type=int, 
                    help='Initial number of grid cells in the x direction, default is 4.')
parser.add_argument('--ny',
                    default=16, type=int,
                    help='Initial number of grid cells in the y direction, default is 4.')
args = parser.parse_args()

bm.set_backend(args.backend)
pde = BoxDomainTriData2D()
nx, ny = args.nx, args.ny
extent = pde.domain()
#mesh = QuadrangleMesh.from_box(box=extent, nx=nx, ny=ny)

p = args.degree
mm = 1e-03
#包壳厚度
w = 0.15 * mm
#半圆半径
R1 = 0.5 * mm
#四分之一圆半径
R2 = 1.0 * mm
#连接处直线段
L = 0.575 * mm
#内部单元大小
h = 0.5 * mm
#棒长
l = 20 * mm
#螺距
#p = 40 * mm
from app.FuelRodSim.fuel_rod_mesher import FuelRodMesher
mesher = FuelRodMesher(R1,R2,L,w,h,meshtype='segmented',modeltype='2D')
mesh = mesher.get_mesh
ficdx,cacidx = mesher.get_2D_fcidx_cacidx()
cnidx,bdnidx = mesher.get_2D_cnidx_bdnidx()
tmr = timer("FEM Solver")
next(tmr)

maxit = 4
errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
errorType = ['$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$']
NDof = bm.zeros(maxit, dtype=bm.int32)
for i in range(maxit):

    space = LagrangeFESpace(mesh, p=p, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
    gdof = space.number_of_global_dofs()
    NDof[i] = tensor_space.number_of_global_dofs()

    pfcm = LinearElasticMaterial(name='E1nu0.3', 
                                                elastic_modulus=1, poisson_ratio=0.3, 
                                                hypo='plane_strain', device=bm.get_device(mesh))
    tmr.send('material')
    integrator_K = LinearElasticIntegrator(material=pfcm, q=tensor_space.p+3)
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_K)
    K = bform.assembly(format='csr')
    # tmr.send('stiffness assembly')
    
    integrator_F = VectorSourceIntegrator(source=pde.source, q=tensor_space.p+3)
    lform = LinearForm(tensor_space)    
    lform.add_integrator(integrator_F)
    F = lform.assembly()
    tmr.send('source assembly')

    dbc = DirichletBC(space=tensor_space, 
                    gd=pde.dirichlet, 
                    threshold=None, 
                    method='interp')
    K, F = dbc.apply(A=K, f=F, uh=None, gd=pde.dirichlet, check=True)
    # uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), 
    #                 dtype=bm.float64, device=bm.get_device(mesh))
    # uh_bd, isDDof = tensor_space.boundary_interpolate(gD=pde.dirichlet, uh=uh_bd, 
    #                                                 threshold=None, method='interp')
    # F = F - K.matmul(uh_bd)
    # F = bm.set_at(F, isDDof, uh_bd[isDDof])
    # K = dbc.apply_matrix(matrix=K, check=True)
    tmr.send('boundary')

    uh = tensor_space.function()

    if args.solver == 'cg':
        uh[:] = cg(K, F, maxiter=1000, atol=1e-14, rtol=1e-14)
    elif args.solver == 'spsolve':
        uh[:] = spsolve(K, F, solver='mumps')
    tmr.send('solve({})'.format(args.solver))
    u_exact = tensor_space.interpolate(pde.solution)
    errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh.astype(bm.float64) - u_exact.astype(bm.float64))**2 * (1 / NDof[i])))

    errorMatrix[1, i] = mesh.error(u=uh, v=pde.solution, q=tensor_space.p+3, power=2)

    if i < maxit-1:
        mesh.uniform_refine()
import os
output = './mesh_linear/'
if not os.path.exists(output):
    os.makedirs(output)
fname = os.path.join(output, 'linear_elastic.vtu')
dofs = space.number_of_global_dofs()
mesh.nodedata['u'] = uh[:dofs]
mesh.nodedata['v'] = uh[-dofs:]
mesh.to_vtk(fname=fname)

print("errorMatrix:\n", errorMatrix)
print("NDof:", NDof)
print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
print("order_L2:\n ", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))
   
   
