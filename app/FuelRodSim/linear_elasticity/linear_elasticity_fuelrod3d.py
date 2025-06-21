from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike

from fealpy.decorator import cartesian

from fealpy.mesh import TetrahedronMesh
from fealpy.fem import DirichletBC    
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.material.elastic_material import LinearElasticMaterial

from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm

from fealpy.decorator import cartesian

from fealpy.sparse import COOTensor

from fealpy.solver import cg,spsolve

from app.soptx.soptx.utils.timer import timer

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
        val = bm.zeros(points.shape, dtype=points.dtype)
        val[..., 0] = 2*x**3 - 3*x*y**2 - 3*x*z**2
        val[..., 1] = 2*y**3 - 3*y*x**2 - 3*y*z**2
        val[..., 2] = 2*z**3 - 3*z*y**2 - 3*z*x**2
        
        return val

    @cartesian
    def source(self, points: TensorLike):
        val = bm.zeros(points.shape, dtype=points.dtype)
        
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
        val = bm.zeros(points.shape, dtype=bm.float64)
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
        val = bm.zeros(points.shape, dtype=bm.float64)

        mu = 1
        val[..., 0] = 200*mu*(x-x**2)**2 * (2*y**3-3*y**2+y) * (2*z**3-3*z**2+z)
        val[..., 1] = -100*mu*(y-y**2)**2 * (2*x**3-3*x**2+x) * (2*z**3-3*z**2+z)
        val[..., 2] = -100*mu*(z-z**2)**2 * (2*y**3-3*y**2+y) * (2*x**3-3*x**2+x)

        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype)

parser = argparse.ArgumentParser(description="TetrahedronMesh 上的任意次 Lagrange 有限元空间的线性弹性问题求解.")
parser.add_argument('--backend', 
                    default='numpy', type=str,
                    help='指定计算的后端类型, 默认为 numpy.')
parser.add_argument('--degree', 
                    default=1, type=int, 
                    help='Lagrange 有限元空间的次数, 默认为 1 次.')
parser.add_argument('--nx', 
                    default=2, type=int, 
                    help='x 方向的初始网格单元数, 默认为 2.')
parser.add_argument('--ny',
                    default=2, type=int,
                    help='y 方向的初始网格单元数, 默认为 2.')
parser.add_argument('--nz',
                    default=2, type=int,
                    help='z 方向的初始网格单元数, 默认为 2.')
args = parser.parse_args()

pde = BoxDomainPolyUnloaded3d()
args = parser.parse_args()

bm.set_backend(args.backend)
nx, ny, nz = args.nx, args.ny, args.nz
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
P = 40 * mm
from app.FuelRodSim.fuel_rod_mesher import FuelRodMesher
from app.FuelRodSim.HeatEquationData import FuelRod3dData 
from app.FuelRodSim.heat_equation_solver import HeatEquationSolver
mesher = FuelRodMesher(R1,R2,L,w,h,l,P,meshtype='segmented',modeltype='3D')
mesh = mesher.get_mesh
ficdx,cacidx = mesher.get_3D_fcidx_cacidx()
cnidx,bdnidx = mesher.get_3D_cnidx_bdnidx()
#mesh = TetrahedronMesh.from_box(box=pde.domain(), nx=nx, ny=ny, nz=nz)
p = args.degree

tmr = timer("FEM Solver")
next(tmr)

maxit = 1
errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
errorType = ['$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$']
NDof = bm.zeros(maxit, dtype=bm.int32)
for i in range(maxit):

    space = LagrangeFESpace(mesh, p=p, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(-1, 3))
    NDof[i] = tensor_space.number_of_global_dofs()

    linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='3D')
    tmr.send('material')

    integrator = LinearElasticIntegrator(material=linear_elastic_material, 
                                        q=tensor_space.p+3)
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator)
    K = bform.assembly()
    tmr.send('stiffness assembly')

    integrator_F = VectorSourceIntegrator(source=pde.source, q=tensor_space.p+3)
    lform = LinearForm(tensor_space)    
    lform.add_integrator(integrator_F)
    F = lform.assembly()
    tmr.send('source assembly')
    """
    uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), dtype=bm.float64)
    uh_bd, isDDof = tensor_space.boundary_interpolate(gd=pde.dirichlet, uh=uh_bd, threshold=None)

    F = F - K.matmul(uh_bd)
    F[isDDof] = uh_bd[isDDof]

    indices = K.tocoo.indices()
    new_values = bm.copy(K.values())
    IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
    new_values[IDX] = 0

    K = COOTensor(indices, new_values, K.sparse_shape)
    index, = bm.nonzero(isDDof)
    one_values = bm.ones(len(index), **K.values_context())
    one_indices = bm.stack([index, index], axis=0)
    K1 = COOTensor(one_indices, one_values, K.sparse_shape)
    K = K.add(K1).coalesce()
    """
    dbc = DirichletBC(space=tensor_space, 
                    gd=pde.dirichlet, 
                    threshold=None, 
                    method='interp')
    K, F = dbc.apply(A=K, f=F, uh=None, gd=pde.dirichlet, check=True)
    tmr.send('boundary')

    uh = tensor_space.function()

    uh[:] = cg(K, F, maxiter=1000, atol=1e-14, rtol=1e-14)
    tmr.send('solve({})')
    
    tmr.send(None)

    u_exact = tensor_space.interpolate(pde.solution)
    # errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh - u_exact)**2 * (1 / NDof[i])))
    # errorMatrix[2, i] = bm.sqrt(bm.sum(bm.abs(uh[isDDof] - u_exact[isDDof])**2 * (1 / NDof[i])))
    errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh.astype(bm.float64) - u_exact.astype(bm.float64))**2 * (1 / NDof[i])))

    errorMatrix[1, i] = mesh.error(u=uh, v=pde.solution, q=tensor_space.p+3, power=2)
    # print("errorMatrix:", errorMatrix)

    if i < maxit-1:
        mesh.uniform_refine()
import os
output = './mesh_rodlinear3d/'
if not os.path.exists(output):
    os.makedirs(output)
fname = os.path.join(output, 'linear_elastic.vtu')
dofs = space.number_of_global_dofs()
mesh.nodedata['u'] = uh[:dofs]
mesh.nodedata['v'] = uh[-2*dofs:-dofs+1]
mesh.nodedata['z'] = uh[-dofs:]
mesh.to_vtk(fname=fname)

print("errorMatrix:\n", errorMatrix)
print("NDof:", NDof)
print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
print("order_L2:\n ", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))