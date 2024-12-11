from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh, TetrahedronMesh
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.sparse import COOTensor, CSRTensor
from fealpy.solver import cg, spsolve

from app.gearx.utils import *
    
bm.set_backend('numpy')

def compute_equivalent_strain(strain, nu):
    exx = strain[..., 0, 0]
    eyy = strain[..., 1, 1]
    ezz = strain[..., 2, 2]
    gamma_xy = strain[..., 0, 1]
    gamma_yz = strain[..., 1, 2]
    gamma_xz = strain[..., 0, 2]
    
    d1 = exx - eyy
    d2 = eyy - ezz
    d3 = ezz - exx
    
    equiv_strain = (d1**2 + d2**2 + d3**2 + 6.0 * (gamma_xy**2 + gamma_yz**2 + gamma_xz**2))
    
    # equiv_strain = bm.sqrt(equiv_strain / 2.0) / (1.0 + nu)
    equiv_strain = bm.sqrt(equiv_strain / 2.0) / (1.0)
    
    return equiv_strain

def compute_equivalent_stress(stress, nu):
    sxx = stress[..., 0, 0]
    syy = stress[..., 1, 1]
    szz = stress[..., 2, 2]
    sxy = stress[..., 0, 1]
    syz = stress[..., 1, 2]
    sxz = stress[..., 0, 2]
    
    d1 = sxx - syy
    d2 = syy - szz
    d3 = szz - sxx
    
    equiv_stress = (d1**2 + d2**2 + d3**2 + 6.0 * (sxy**2 + syz**2 + sxz**2))

    equiv_stress = bm.sqrt(equiv_stress / 2.0)
    
    return equiv_stress

class BoxDomainLinear3d():
    def __init__(self):
        self.eps = 1e-12

    def domain(self):
        return [0, 1, 0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike):
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))
        
        return val

    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))
        
        val[..., 0] = 1e-3 * (2*x + y + z) / 2
        val[..., 1] = 1e-3 * (x + 2*y + z) / 2
        val[..., 2] = 1e-3 * (x + y + 2*z) / 2

        return val

    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:

        result = self.solution(points)

        return result

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
# mesh = HexahedronMesh(node, cell)

nx, ny, nz = 2, 2, 2 
# mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], 
#                             nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))
mesh = TetrahedronMesh.from_box([0, 1, 0, 1, 0, 1], 
                            nx=nx, ny=ny, nz=ny, device=bm.get_device('cpu'))

GD = mesh.geo_dimension()
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
print(f"NN = {NN} NC = {NC}")
cm = mesh.cell_volume()
node = mesh.entity('node')
cell = mesh.entity('cell')

p = 1
q = p+1
space = LagrangeFESpace(mesh, p=p, ctype='C')
sgdof = space.number_of_global_dofs()
print(f"sgdof: {sgdof}")
# tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority
tensor_space = TensorFunctionSpace(space, shape=(-1, 3)) # gd_priority
tldof = tensor_space.number_of_local_dofs()
tgdof = tensor_space.number_of_global_dofs()
print(f"tgdof: {tgdof}")
pde = BoxDomainLinear3d()

# 刚度矩阵
E = 2.1e5
nu = 0.3
# E = 1e6
# nu = 0.25
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=q, method='voigt')
KE = integrator_K.voigt_assembly(space=tensor_space)
KE0 = KE[0]
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')
Kdense = K.to_dense()
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
print(f"K.shape = {K.shape}")
print(f"Matrix norm before dc: {K_norm:.6f}")

# 载荷向量
integrator_F = VectorSourceIntegrator(source=pde.source, q=q)
lform = LinearForm(tensor_space)    
lform.add_integrator(integrator_F)
F = lform.assembly()
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"F.shape = {F.shape}")
print(f"Load vector norm before dc: {F_norm:.6f}")

# 导入 inp 文件
u = lambda p: (10e-3*(2*p[..., 0]+p[..., 1]+p[..., 2])/2).reshape(-1, 1)
v = lambda p: (10e-3*(p[..., 0]+2*p[..., 1]+p[..., 2])/2).reshape(-1, 1)
w = lambda p: (10e-3*(p[..., 0]+p[..., 1]+2*p[..., 2])/2).reshape(-1, 1)
isBdNode = space.is_boundary_dof(threshold=None)
boundary_nodes_idx = bm.where(isBdNode)[0]
boundary_nodes = node[boundary_nodes_idx]
boundary_nodes_u = bm.concatenate([u(boundary_nodes), v(boundary_nodes), w(boundary_nodes)], axis=1)
export_to_inp_by_u(filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/linear_fealpy.inp', 
              nodes=node, elements=cell, 
              boundary_nodes_idx=boundary_nodes_idx, boundary_nodes_u=boundary_nodes_u,
              young_modulus=E, poisson_ratio=nu, density=7.85e-9, used_app='ansys')

# 边界条件处理
uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), 
                dtype=bm.float64, device=bm.get_device(mesh))
uh_bd, isDDof = tensor_space.boundary_interpolate(gd=pde.dirichlet, 
                                                uh=uh_bd, threshold=None, method='interp')
print(f"isDDof: {isDDof.shape} :\n {isDDof}")
print(f"uh_bd: {uh_bd.shape} :\n {uh_bd}")
F = F - K.matmul(uh_bd)
F = bm.set_at(F, isDDof, uh_bd[isDDof])

dbc = DirichletBC(space=tensor_space, 
                gd=pde.dirichlet, 
                threshold=None, 
                method='interp')
K = dbc.apply_matrix(matrix=K, check=True)
Kdense1 = K.to_dense()
# 求解
uh = tensor_space.function()
# uh[:] = cg(K, F, maxiter=1000, atol=1e-8, rtol=1e-8)
uh[:] = spsolve(K, F, solver="mumps")
print(f"uh: {uh.shape}:\n {uh[:]}")
# print(f"uh_inner {uh[~isDDof].shape}:\n {uh[~isDDof]}")
# print(f"uh_bounder {uh[isDDof].shape}:\n {uh[isDDof]}")
u_exact = tensor_space.interpolate(pde.solution)
print(f"u_exact: {u_exact[:].shape}:\n {u_exact[:]}")
# print(f"u_exact_inner {u_exact[~isDDof].shape}:\n {u_exact[~isDDof]}")
# print(f"u_exact_bounder {u_exact[isDDof].shape}:\n {u_exact[isDDof]}")

if tensor_space.dof_priority:
    uh_show = uh.reshape(GD, NN).T
else:
    uh_show = uh.reshape(NN, GD)
uh_x = uh_show[:, 0]
uh_y = uh_show[:, 1]
uh_z = uh_show[:, 2]
print(f"uh_x: {uh_x.shape}:\n {uh_x}")
print(f"uh_y: {uh_y.shape}:\n {uh_y}")
print(f"uh_z: {uh_z.shape}:\n {uh_z}")
uh_magnitude = bm.linalg.norm(uh_show, axis=1)

uh_cell = bm.zeros((NC, tldof)) # (NC, tldof)
cell2tdof = tensor_space.cell_to_dof()
for c in range(NC):
    uh_cell[c] = uh[cell2tdof[c]]

# 单元积分点处的位移梯度
q = 2
qf = mesh.quadrature_formula(q)
bcs_quadrature, ws = qf.get_quadrature_points_and_weights()
tgphi = tensor_space.grad_basis(bcs_quadrature)             # (NC, NQ, tldof, GD, GD)
tgrad = bm.einsum('cqimn, ci -> cqmn', tgphi, uh_cell)      # (NC, NQ, GD, GD)

# 应变张量
strain = (tgrad + bm.transpose(tgrad, (0, 1, 3, 2))) / 2    # (NC, NQ, GD, GD)
strain_xx = strain[..., 0, 0] # (NC, NQ)
strain_yy = strain[..., 1, 1] # (NC, NQ)
strain_zz = strain[..., 2, 2] # (NC, NQ)
strain_xy = strain[..., 0, 1] # (NC, NQ)
strain_yz = strain[..., 1, 2] # (NC, NQ)
strain_xz = strain[..., 0, 2] # (NC, NQ)

# 应力张量
trace_e = bm.einsum("...ii", strain) # (NC, NQ)
I = bm.eye(GD, dtype=bm.float64)
stress = 2 * mu * strain + lam * trace_e[..., None, None] * I # (NC, NQ, GD, GD)
stress_xx = stress[..., 0, 0] # (NC, 8)
stress_yy = stress[..., 1, 1] # (NC, 8)
stress_zz = stress[..., 2, 2] # (NC, 8)
stress_xy = stress[..., 0, 1] # (NC, 8)
stress_yz = stress[..., 1, 2] # (NC, 8)
stress_xz = stress[..., 0, 2] # (NC, 8)

# 后处理
if tensor_space.dof_priority:
    uh_show = uh.reshape(GD, NN).T
else:
    uh_show = uh.reshape(NN, GD)
uh_magnitude = bm.linalg.norm(uh_show, axis=1)
mesh.nodedata['uh'] = uh_show[:]
mesh.nodedata['uh_magnitude'] = uh_magnitude[:]
mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gear_box_fealpy.vtu')


# 验证收敛阶
maxit = 4
errorType = ['$|| u_{bd} - uh_{bd}$', '$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$']
errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
NDof = bm.zeros(maxit, dtype=bm.int32)
for i in range(maxit):
    space = LagrangeFESpace(mesh, p=p, ctype='C')
    # tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority
    tensor_space = TensorFunctionSpace(space, shape=(-1, 3)) # gd_priority

    NDof[i] = tensor_space.number_of_global_dofs()

    linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))

    integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=q, method='voigt')
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_K)
    K = bform.assembly(format='csr')

    integrator_F = VectorSourceIntegrator(source=pde.source, q=q)
    lform = LinearForm(tensor_space)    
    lform.add_integrator(integrator_F)
    F = lform.assembly()

    dbc = DirichletBC(space=tensor_space, 
                    gd=pde.dirichlet, 
                    threshold=None, 
                    method='interp')
    uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), 
                    dtype=bm.float64, device=bm.get_device(mesh))
    uh_bd, isDDof = tensor_space.boundary_interpolate(gd=pde.dirichlet, 
                                                    uh=uh_bd, threshold=None, method='interp')
    F = F - K.matmul(uh_bd)
    F = bm.set_at(F, isDDof, uh_bd[isDDof])

    K = dbc.apply_matrix(matrix=K, check=True)

    uh = tensor_space.function()
    # uh[:] = cg(K, F, maxiter=1000, atol=1e-8, rtol=1e-8)
    uh[:] = spsolve(K, F, solver="mumps")

    u_exact = tensor_space.interpolate(pde.solution)
    errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh[:] - u_exact)**2 * (1 / NDof[i])))
    errorMatrix[1, i] = mesh.error(u=uh, v=pde.solution, q=tensor_space.p+3, power=2)
    errorMatrix[2, i] = bm.sqrt(bm.sum(bm.abs(uh[isDDof] - u_exact[isDDof])**2 * (1 / NDof[i])))

    if i < maxit-1:
        mesh.uniform_refine()

print("errorMatrix:\n", errorType, "\n", errorMatrix)
print("NDof:", NDof)
print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
print("order_L2:\n ", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))
print("----------------------")

