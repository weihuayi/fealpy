from fealpy.backend import backend_manager as bm

from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.mesh import QuadrangleMesh,TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.decorator import cartesian
from fealpy.solver import cg, spsolve

from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.material.elastico_plastic_material import PlasticMaterial
from fealpy.fem.elasticoplastic_integrator import TransitionElasticIntegrator

import argparse
# 平面应变问题
class CantileverBeamData2D():

    def domain(self):
        # 定义悬臂梁的几何尺寸
        return [0, 1, 0, 0.25]  # 长度为1 dm，高度为0.25 dm
    
    def durtion(self):
        # 时间区间
        return [0, 1]

    @cartesian
    def source(self, points: TensorLike, index=None, time: float = 0.0) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        # 假设载荷随时间变化，t为当前时间步
        val[..., 1] = -1.7 * (1 + 0.1 * time)  # 载荷随时间增大

        return val

    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:
        # 固定端边界条件（假设左端固定）
        x = points[..., 0]
        fixed_mask = x < 1e-10  # 左端固定
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[fixed_mask] = [0, 0]  # 位移为零
        
        return val

    

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
                    default=20, type=int, 
                    help='Initial number of grid cells in the x direction, default is 4.')
parser.add_argument('--ny',
                    default=20, type=int,
                    help='Initial number of grid cells in the y direction, default is 4.')
args = parser.parse_args()

bm.set_backend(args.backend)
pde = CantileverBeamData2D()
nx, ny = args.nx, args.ny
extent = pde.domain()
nt = 10
dt = pde.durtion()[1] / nt
mesh = TriangleMesh.from_box(box=extent, nx=nx, ny=ny)

p = args.degree
space = LagrangeFESpace(mesh, p=p, ctype='C')
tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
gdof = space.number_of_global_dofs()

pfcm = LinearElasticMaterial(name='E1nu0.3', 
                                            elastic_modulus=1, poisson_ratio=0.3, 
                                            hypo='plane_strain', device=bm.get_device(mesh))
qf = mesh.quadrature_formula(q=tensor_space.p+3)
bcs, ws = qf.get_quadrature_points_and_weights()
# 在时间循环外部初始化历史变量
NC = mesh.number_of_cells()
NQ = len(ws)
sigma_eff_prev = bm.zeros((NC, NQ), dtype=bm.float64)  # 存储前一时刻等效应力
plastic_flag = bm.zeros((NC, NQ), dtype=bool)          # 塑性状态标记
D_ep_prev = pfcm.elastic_matrix()                      # 初始弹性矩阵

for step in range(nt):
    if step == 0:
        integrator_K = LinearElasticIntegrator(material=pfcm, q=tensor_space.p+3)
        bform = BilinearForm(tensor_space)
        bform.add_integrator(integrator_K)
        K = bform.assembly(format='csr')
    if step > 0:
         # 创建新积分器并组装刚度矩阵
        integrator_K = TransitionElasticIntegrator(D_ep, material=pfcm, q=tensor_space.p+3,method='voigt')
        bform = BilinearForm(tensor_space)
        bform.add_integrator(integrator_K)
        K = bform.assembly(format='csr')
    # tmr.send('stiffness assembly')
    lform = LinearForm(tensor_space)  
    from fealpy.decorator import cartesian
    @cartesian
    def coef(p):
        time = step * dt
        val = pde.source(p, time)
        return val
    lform.add_integrator(VectorSourceIntegrator(coef))
    F = lform.assembly()
    dbc = DirichletBC(space=tensor_space, 
                    gd=pde.dirichlet, 
                    threshold=None, 
                    method='interp')
    K, F = dbc.apply(A=K, f=F, uh=None, gd=pde.dirichlet, check=True)
    uh = tensor_space.function()
    if args.solver == 'cg':
        uh[:] = cg(K, F, maxiter=1000, atol=1e-14, rtol=1e-14)
    elif args.solver == 'spsolve':
        uh[:] = spsolve(K, F, solver='mumps')
    
    # 计算单元上积分点的应力、应变和等效应力
    gphi = space.grad_basis(bcs)
    NC = mesh.number_of_cells()
    tldof = tensor_space.number_of_local_dofs()
    cell2tdof = tensor_space.cell_to_dof()
    uh = bm.array(uh)  
    uh_cell = bm.zeros((NC, tldof)) # (NC, tldof)
    for c in range(NC):
        uh_cell[c] = uh[cell2tdof[c]]
    D = pfcm.elastic_matrix()
    B = pfcm.strain_matrix(True, gphi)
    # 计算应变和应力 剪应变和剪应力需要除2
    strain  = bm.einsum('ijkl,il->ijk', B, uh_cell)#(NC,NQ,3)
    sigma  =  bm.einsum('ijkl,ijk->ijl', D, strain)# (NC, 3)
    # 计算应变
    strain[..., 2] /= 2
    # 计算应力
    sigma[..., 2] /= 2
    # 计算等效应力
    sigma_00 = sigma[..., 0]  # 第一个分量 sigma_00
    sigma_11 = sigma[..., 1]  # 第二个分量 sigma_11
    sigma_01 = sigma[..., 2]  # 第三个分量 sigma_01

    # 根据 von Mises 应力公式计算等效应力
    sigma_eff = bm.sqrt(0.5 * ((sigma_00 - sigma_11)**2 + 
                            (sigma_00 - sigma_01)**2 + 
                            (sigma_11 - sigma_01)**2))
    #print(sigma_eff)
    # 判断等效应力是否超过材料的屈服应力
    yield_stress = 0.3
    epfcm = PlasticMaterial(name='E1nu0.3', elastic_modulus=1, poisson_ratio=0.3, 
                                hypo='plane_strain', yield_stress=yield_stress, device=bm.get_device(mesh))
     # 计算当前增量步等效应力增量
    delta_sigma_eff = sigma_eff - sigma_eff_prev
    # 计算达到屈服需要的应力增量
    delta_sigma_needed = yield_stress - sigma_eff_prev
    # 计算加权系数m
    m = delta_sigma_needed / (delta_sigma_eff + 1e-12)  # 防止除零
    print(m)
    # 确定材料状态
    fully_plastic = plastic_flag | (delta_sigma_eff == 0)
    elastic_mask = m >= 1.0
    transition_mask = (m > 0) & (m < 1) & (~fully_plastic)
    new_plastic_mask = (delta_sigma_eff > 0) & (sigma_eff > yield_stress)
    # 更新塑性标志
    plastic_flag = fully_plastic | new_plastic_mask
    print(plastic_flag)
    # 计算混合弹塑性矩阵
    De = pfcm.elastic_matrix()  # (1,1,N,N)
    Dp = epfcm.elastico_plastic_matrix(stress=sigma)  # (NC,NQ,N,N)
    # 扩展弹性矩阵到相同维度
    De_exp = bm.broadcast_to(De, Dp.shape)
    # 计算过渡矩阵
    m_exp = m[..., None, None]  # (NC,NQ,1,1)
    D_ep = m_exp * De_exp + (1 - m_exp) * Dp
    # 处理不同状态
    D_ep = bm.where(elastic_mask[..., None, None], De_exp, D_ep)
    D_ep = bm.where(plastic_flag[..., None, None], Dp, D_ep)
    # 更新历史变量
    sigma_eff_prev = sigma_eff.copy()
    
    
    

'''
import os
output = './mesh_linear/'
if not os.path.exists(output):
    os.makedirs(output)
fname = os.path.join(output, 'linear_elastic.vtu')
dofs = space.number_of_global_dofs()
mesh.nodedata['u'] = uh[:dofs]
mesh.nodedata['v'] = uh[-dofs:]
mesh.to_vtk(fname=fname)
'''


   
