from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.sparse import COOTensor
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.typing import TensorLike
from fealpy.solver import cg, spsolve

import pickle
from app.gearx.gear import ExternalGear, InternalGear
from app.gearx.utils import *

from fealpy.mesh import HexahedronMesh

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

with open('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/external_gear_data_part.pkl', 'rb') as f:
    data = pickle.load(f)

# Ansys 位移结果
u_x_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/u_x_100.txt', 
                                skiprows=1, usecols=1), dtype=bm.float64)
u_y_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/u_y_100.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
u_z_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/u_z_100.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
uh_ansys_show = bm.stack([u_x_ansys, u_y_ansys, u_z_ansys], axis=1)  # (NN, GD)

# Ansys 应变结果     # (NN, )
strain_xx_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/ns_x.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
strain_yy_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/ns_y.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
strain_zz_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/ns_z.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
strain_xy_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/ns_xy.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
strain_yz_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/ns_yz.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
strain_xz_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/ns_xz.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)

# Ansys 节点等效应力 # (NN)
nodal_equiv_stress_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/es_100.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)

external_gear = data['external_gear']
hex_mesh = data['hex_mesh']
helix_node = data['helix_node']
is_inner_node = data['is_inner_node']

hex_cell = hex_mesh.cell
hex_node = hex_mesh.node
mesh = HexahedronMesh(hex_node, hex_cell)

GD = mesh.geo_dimension()   
NC = mesh.number_of_cells()
print(f"NC: {NC}")
NN = mesh.number_of_nodes()
print(f"NN: {NN}")
node = mesh.entity('node')
cell = mesh.entity('cell')

# 节点载荷
load_values = 100 * bm.array([50.0, 60.0, 79.0, 78.0, 87.0, 95.0, 102.0, 109.0, 114.0,
                        119.0, 123.0, 127.0, 129.0, 130.0, 131.0], dtype=bm.float64) # (15, )

n = 15
helix_d = bm.linspace(external_gear.d, external_gear.effective_da, n)
helix_width = bm.linspace(0, external_gear.tooth_width, n)
helix_node = external_gear.cylindrical_to_cartesian(helix_d, helix_width)

target_cell_idx = bm.zeros(n, bm.int32)
face_normal = bm.zeros((n, 3), bm.float64)
parameters = bm.zeros((n, 3), bm.float64)
for i, t_node in enumerate(helix_node):
    target_cell_idx[i], face_normal[i], parameters[i] = external_gear.find_node_location_kd_tree(t_node)

average_normal = bm.mean(face_normal, axis=0)
average_normal /= bm.linalg.norm(average_normal)

threshold = 0.1
for i in range(len(face_normal)):
    deviation = np.linalg.norm(face_normal[i] - average_normal)
    if deviation > threshold:
        face_normal[i] = average_normal

P = bm.einsum('p, pd -> pd', load_values, face_normal)  # (15, GD)

u = parameters[..., 0]
v = parameters[..., 1]
w = parameters[..., 2]

bcs_list = [
    (
        bm.tensor([[u, 1 - u]]),
        bm.tensor([[v, 1 - v]]),
        bm.tensor([[w, 1 - w]])
    )
    for u, v, w in zip(u, v, w)
]
p = 1
space = LagrangeFESpace(mesh, p=p, ctype='C')
scalar_gdof = space.number_of_global_dofs()
print(f"gdof: {scalar_gdof}")
cell2dof = space.cell_to_dof()

q = p+1

# tensor_space = TensorFunctionSpace(space, shape=(-1, 3)) # gd_priority
tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority

tgdof = tensor_space.number_of_global_dofs()
print(f"tgdof: {tgdof}")
tldof = tensor_space.number_of_local_dofs()
cell2tdof = tensor_space.cell_to_dof()

load_node_indices = cell[target_cell_idx].flatten() # (15*8, )
# 带有载荷的节点对应的全局自由度编号
dof_indices = bm.stack([scalar_gdof * d + 
                        load_node_indices for d in range(GD)], axis=1)  # (15*8, GD)

phi_loads = []
for bcs in bcs_list:
    phi = tensor_space.basis(bcs)
    phi_loads.append(phi)

phi_loads_array = bm.concatenate(phi_loads, axis=1) # (1, 15, tldof, GD)

FE_load = bm.einsum('pd, cpld -> pl', P, phi_loads_array) # (15, 24)

FE = bm.zeros((NC, tldof), dtype=bm.float64)
FE[target_cell_idx, :] = FE_load[:, :] # (NC, tldof)

F = COOTensor(indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, ))
indices = cell2tdof.reshape(1, -1)
F = F.add(COOTensor(indices, FE.reshape(-1), (tgdof, ))).to_dense() # (tgdof, )

# 从全局载荷向量中提取有载荷节点处的值
F_load_nodes = F[dof_indices] # (15*8, GD)

fixed_node_index = bm.where(is_inner_node)[0]
export_to_inp(filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gear_fealpy.inp', 
              nodes=node, elements=cell, 
              fixed_nodes=fixed_node_index, load_nodes=load_node_indices, loads=F_load_nodes, 
              young_modulus=206e3, poisson_ratio=0.3, density=7.85e-9)

E = 206e3
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=q)
# integrator_K_method = LinearElasticIntegrator(material=linear_elastic_material, q=q, method='voigt')
# KE = integrator_K.assembly(tensor_space)
# KE_method = integrator_K.voigt_assembly(tensor_space)
# error = bm.sum(bm.abs(KE - KE_method))

bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"Matrix norm after dc: {K_norm:.6f}")
print(f"Load vector norm after dc: {F_norm:.6f}")

scalar_is_bd_dof = bm.zeros(scalar_gdof, dtype=bm.bool)
scalar_is_bd_dof[:NN] = is_inner_node
tensor_is_bd_dof = tensor_space.is_boundary_dof(
        threshold=(scalar_is_bd_dof, scalar_is_bd_dof, scalar_is_bd_dof), 
        method='interp')
dbc = DirichletBC(space=tensor_space, 
                    gd=0, 
                    threshold=tensor_is_bd_dof, 
                    method='interp')
K = dbc.apply_matrix(matrix=K, check=True)
uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), dtype=bm.float64, device=bm.get_device(mesh))
# uh_bd, isDDof = tensor_space.boundary_interpolate(gd=0, 
#                                                 uh=uh_bd, 
#                                                 threshold=tensor_is_bd_dof, 
#                                                 method='interp')
isDDof = tensor_is_bd_dof
F = F - K.matmul(uh_bd)
F = bm.set_at(F, isDDof, uh_bd[isDDof])
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"Matrix norm after dc: {K_norm:.6f}")
print(f"Load vector norm after dc: {F_norm:.6f}")

from fealpy import logger
logger.setLevel('INFO')

uh = tensor_space.function()
# uh[:] = cg(K, F, maxiter=10000, atol=1e-8, rtol=1e-8)
uh[:] = spsolve(K, F, solver="mumps")

uh_ansys = tensor_space.function()
if tensor_space.dof_priority:
    uh_ansys[:] = uh_ansys_show.T.flatten()
else:
    uh_ansys[:] = uh_ansys_show.flatten()

# 计算残差向量和范数
residual = K.matmul(uh[:]) - F  # 使用 CSRTensor 的 matmul 方法
residual_norm = bm.sqrt(bm.sum(residual * residual))
print(f"Final residual norm: {residual_norm:.6e}")

if tensor_space.dof_priority:
    uh_show = uh.reshape(GD, NN).T
else:
    uh_show = uh.reshape(NN, GD)
uh_x = uh_show[:, 0]
uh_y = uh_show[:, 1]
uh_z = uh_show[:, 2]

# 位移误差
error_x = uh_x - u_x_ansys # (NN, )
error_y = uh_y - u_y_ansys
error_z = uh_z - u_z_ansys
relative_error_x = bm.linalg.norm(error_x) / (bm.linalg.norm(u_x_ansys)+bm.linalg.norm(uh_x))
relative_error_y = bm.linalg.norm(error_y) / (bm.linalg.norm(u_y_ansys)+bm.linalg.norm(uh_y))
relative_error_z = bm.linalg.norm(error_z) / (bm.linalg.norm(u_z_ansys)+bm.linalg.norm(uh_z))
print(f"Relative error_x: {relative_error_x:.12e}")
print(f"Relative error_y: {relative_error_y:.12e}")
print(f"Relative error_z: {relative_error_z:.12e}")

uh_magnitude = bm.linalg.norm(uh_show, axis=1)

mesh.nodedata['uh'] = uh_show[:]
mesh.nodedata['uh_magnitude'] = uh_magnitude[:]

uh_cell = bm.zeros((NC, tldof)) # (NC, tldof)
uh_cell_ansys = bm.zeros((NC, tldof))
for c in range(NC):
    uh_cell[c] = uh[cell2tdof[c]]
    uh_cell_ansys[c] = uh_ansys[cell2tdof[c]]

# 节点处的位移梯度
# 方法一
bcs1 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64))
p1 = mesh.bc_to_point(bc=bcs1)
bcs2 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64))
p2 = mesh.bc_to_point(bc=bcs2)
bcs3 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64))
p3 = mesh.bc_to_point(bc=bcs3)
bcs4 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64))
p4 = mesh.bc_to_point(bc=bcs4)
bcs5 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64))
p5 = mesh.bc_to_point(bc=bcs5)
bcs6 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64))
p6 = mesh.bc_to_point(bc=bcs6)
bcs7 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64))
p7 = mesh.bc_to_point(bc=bcs7)
bcs8 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64))
p8 = mesh.bc_to_point(bc=bcs8)
tgphi1 = tensor_space.grad_basis(bcs1) # (NC, 1, tldof, GD, GD)
tgphi2 = tensor_space.grad_basis(bcs2) # (NC, 1, tldof, GD, GD)
tgphi3 = tensor_space.grad_basis(bcs3) # (NC, 1, tldof, GD, GD)
tgphi4 = tensor_space.grad_basis(bcs4) # (NC, 1, tldof, GD, GD)
tgphi5 = tensor_space.grad_basis(bcs5) # (NC, 1, tldof, GD, GD)
tgphi6 = tensor_space.grad_basis(bcs6) # (NC, 1, tldof, GD, GD)
tgphi7 = tensor_space.grad_basis(bcs7) # (NC, 1, tldof, GD, GD)
tgphi8 = tensor_space.grad_basis(bcs8) # (NC, 1, tldof, GD, GD)
tgphi = bm.concatenate([tgphi1, tgphi2, tgphi3, tgphi4, 
                        tgphi5, tgphi6, tgphi7, tgphi8], axis=1) # (NC, 8, tldof, GD, GD)
# # 方法二
# # 插值点的多重指标
# shape = (p+1, p+1, p+1)
# mi = bm.arange(p+1, device=bm.get_device(cell))
# multiIndex0 = bm.broadcast_to(mi[:, None, None], shape).reshape(-1, 1)
# multiIndex1 = bm.broadcast_to(mi[None, :, None], shape).reshape(-1, 1)
# multiIndex2 = bm.broadcast_to(mi[None, None, :], shape).reshape(-1, 1)
# multiIndex = bm.concatenate([multiIndex0, multiIndex1, multiIndex2], axis=-1)
# # 多重指标的映射
# multiIndex_map = mesh.multi_index_matrix(p=p, etype=1)
# # 插值点的重心坐标
# barycenter = multiIndex_map[multiIndex].astype(bm.float64)
# bcs = (barycenter[:, 0, :], barycenter[:, 1, :], barycenter[:, 2, :])

# tgphi_list = []
# for i in range(barycenter.shape[0]):
#     bc_i = (
#         bcs[0][i].reshape(1, -1),  # (1, 2)
#         bcs[1][i].reshape(1, -1),  # (1, 2)
#         bcs[2][i].reshape(1, -1)   # (1, 2)
#     )
#     tgphi_i = tensor_space.grad_basis(bc_i)  # (NC, 1, tldof, GD, GD)
#     tgphi_list.append(tgphi_i)
# tgphi = bm.concatenate(tgphi_list, axis=1)  # (NC, 8, tldof, GD, GD)
tgrad = bm.einsum('cqimn, ci -> cqmn', tgphi, uh_cell)      # (NC, 8, GD, GD)

# 应变张量
strain = (tgrad + bm.transpose(tgrad, (0, 1, 3, 2))) / 2 # (NC, 8, GD, GD)
strain_xx = strain[..., 0, 0] # (NC, 8)
strain_yy = strain[..., 1, 1] # (NC, 8)
strain_zz = strain[..., 2, 2] # (NC, 8)
strain_xy = strain[..., 0, 1] # (NC, 8)
strain_yz = strain[..., 1, 2] # (NC, 8)
strain_xz = strain[..., 0, 2] # (NC, 8)

# 基于 Ansys 中的位移的应变张量
tgrad_ansys = bm.einsum('cqimn, ci -> cqmn', tgphi, uh_cell_ansys)      # (NC, 8, GD, GD)
strain_ansys_u = (tgrad_ansys + bm.transpose(tgrad_ansys, (0, 1, 3, 2))) / 2 # (NC, 8, GD, GD)
strain_xx_ansys_u = strain_ansys_u[..., 0, 0] # (NC, 8)
strain_yy_ansys_u = strain_ansys_u[..., 1, 1] # (NC, 8)
strain_zz_ansys_u = strain_ansys_u[..., 2, 2] # (NC, 8)
strain_xy_ansys_u = strain_ansys_u[..., 0, 1] # (NC, 8)
strain_yz_ansys_u = strain_ansys_u[..., 1, 2] # (NC, 8)
strain_xz_ansys_u = strain_ansys_u[..., 0, 2] # (NC, 8)

# 节点计算次数
num_count = bm.zeros(NN, dtype=bm.int32)
bm.add_at(num_count, cell2dof.flatten(), 1)

# 节点应变张量
nodal_strain_xx = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xx, cell2dof.flatten(), strain_xx.flatten())
nodal_strain_xx /= num_count
nodal_strain_yy = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_yy, cell2dof.flatten(), strain_yy.flatten())
nodal_strain_yy /= num_count
nodal_strain_zz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_zz, cell2dof.flatten(), strain_zz.flatten())
nodal_strain_zz /= num_count
nodal_strain_xy = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xy, cell2dof.flatten(), strain_xy.flatten())
nodal_strain_xy /= num_count
nodal_strain_yz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_yz, cell2dof.flatten(), strain_yz.flatten())
nodal_strain_yz /= num_count
nodal_strain_xz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xz, cell2dof.flatten(), strain_xz.flatten())
nodal_strain_xz /= num_count

# 基于 Ansys 中的位移的节点应变张量
nodal_strain_xx_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xx_ansys_u, cell2dof.flatten(), strain_xx_ansys_u.flatten())
nodal_strain_xx_ansys_u /= num_count
nodal_strain_yy_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_yy_ansys_u, cell2dof.flatten(), strain_yy_ansys_u.flatten())
nodal_strain_yy_ansys_u /= num_count
nodal_strain_zz_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_zz_ansys_u, cell2dof.flatten(), strain_zz_ansys_u.flatten())
nodal_strain_zz_ansys_u /= num_count
nodal_strain_xy_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xy_ansys_u, cell2dof.flatten(), strain_xy_ansys_u.flatten())
nodal_strain_xy_ansys_u /= num_count
nodal_strain_yz_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_yz_ansys_u, cell2dof.flatten(), strain_yz_ansys_u.flatten())
nodal_strain_yz_ansys_u /= num_count
nodal_strain_xz_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xz_ansys_u, cell2dof.flatten(), strain_xz_ansys_u.flatten())
nodal_strain_xz_ansys_u /= num_count

# 基于 Ansys 中位移的节点应变张量和 Ansys 中节点应变张量的误差
error_s_xx_u = nodal_strain_xx_ansys_u - strain_xx_ansys
error_s_yy_u = nodal_strain_yy_ansys_u - strain_yy_ansys
error_s_zz_u = nodal_strain_zz_ansys_u - strain_zz_ansys
error_s_xy_u = nodal_strain_xy_ansys_u - strain_xy_ansys
error_s_yz_u = nodal_strain_yz_ansys_u - strain_yz_ansys
error_s_xz_u = nodal_strain_xz_ansys_u - strain_xz_ansys
relative_error_s_xx_u = bm.linalg.norm(error_s_xx_u) / (
                            bm.linalg.norm(nodal_strain_xx_ansys_u)+bm.linalg.norm(strain_xx_ansys))
relative_error_s_yy_u = bm.linalg.norm(error_s_yy_u) / (
                            bm.linalg.norm(nodal_strain_yy_ansys_u)+bm.linalg.norm(strain_yy_ansys))
relative_error_s_zz_u = bm.linalg.norm(error_s_zz_u) / (
                            bm.linalg.norm(nodal_strain_zz_ansys_u)+bm.linalg.norm(strain_zz_ansys))
relative_error_s_xy_u = bm.linalg.norm(error_s_xy_u) / (
                            bm.linalg.norm(nodal_strain_xy_ansys_u)+bm.linalg.norm(strain_xy_ansys))
relative_error_s_yz_u = bm.linalg.norm(error_s_yz_u) / (
                            bm.linalg.norm(nodal_strain_yz_ansys_u)+bm.linalg.norm(strain_yz_ansys))
relative_error_s_xz_u = bm.linalg.norm(error_s_xz_u) / (
                            bm.linalg.norm(nodal_strain_xz_ansys_u)+bm.linalg.norm(strain_xz_ansys))
print("基于 Ansys 中位移的节点应变张量和 Ansys 中节点应变张量的相对误差")
print(f"Relative error_s_xx_1: {relative_error_s_xx_u:.12e}")
print(f"Relative error_s_yy_1: {relative_error_s_yy_u:.12e}")
print(f"Relative error_s_zz_1: {relative_error_s_zz_u:.12e}")
print(f"Relative error_s_xy_1: {relative_error_s_xy_u:.12e}")
print(f"Relative error_s_yz_1: {relative_error_s_yz_u:.12e}")
print(f"Relative error_s_xz_1: {relative_error_s_xz_u:.12e}")

# 节点应变张量和 Ansys 中节点应变张量的误差
error_s_xx = nodal_strain_xx - strain_xx_ansys
error_s_yy = nodal_strain_yy - strain_yy_ansys
error_s_zz = nodal_strain_zz - strain_zz_ansys
error_s_xy = nodal_strain_xy - strain_xy_ansys
error_s_yz = nodal_strain_yz - strain_yz_ansys
error_s_xz = nodal_strain_xz - strain_xz_ansys
relative_error_s_xx = bm.linalg.norm(error_s_xx) / (bm.linalg.norm(nodal_strain_xx)+bm.linalg.norm(strain_xx_ansys))
relative_error_s_yy = bm.linalg.norm(error_s_yy) / (bm.linalg.norm(nodal_strain_yy)+bm.linalg.norm(strain_yy_ansys))
relative_error_s_zz = bm.linalg.norm(error_s_zz) / (bm.linalg.norm(nodal_strain_zz)+bm.linalg.norm(strain_zz_ansys))
relative_error_s_xy = bm.linalg.norm(error_s_xy) / (bm.linalg.norm(nodal_strain_xy)+bm.linalg.norm(strain_xy_ansys))
relative_error_s_yz = bm.linalg.norm(error_s_yz) / (bm.linalg.norm(nodal_strain_yz)+bm.linalg.norm(strain_yz_ansys))
relative_error_s_xz = bm.linalg.norm(error_s_xz) / (bm.linalg.norm(nodal_strain_xz)+bm.linalg.norm(strain_xz_ansys))
print("节点应变张量和 Ansys 中节点应变张量的相对误差")
print(f"Relative error_s_xx: {relative_error_s_xx:.12e}")
print(f"Relative error_s_yy: {relative_error_s_yy:.12e}")
print(f"Relative error_s_zz: {relative_error_s_zz:.12e}")
print(f"Relative error_s_xy: {relative_error_s_xy:.12e}")
print(f"Relative error_s_yz: {relative_error_s_yz:.12e}")
print(f"Relative error_s_xz: {relative_error_s_xz:.12e}")

# 应力张量
trace_e = bm.einsum("...ii", strain) # (NC, 8)
I = bm.eye(GD, dtype=bm.float64)
stress = 2 * mu * strain + lam * trace_e[..., None, None] * I # (NC, 8, GD, GD)
stress_xx = stress[..., 0, 0] # (NC, 8)
stress_yy = stress[..., 1, 1] # (NC, 8)
stress_zz = stress[..., 2, 2] # (NC, 8)
stress_xy = stress[..., 0, 1] # (NC, 8)
stress_yz = stress[..., 1, 2] # (NC, 8)
stress_xz = stress[..., 0, 2] # (NC, 8)

# 节点应力张量
nodal_stress_xx = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_xx, cell2dof.flatten(), stress_xx.flatten())
nodal_stress_xx = nodal_stress_xx / num_count
nodal_stress_yy = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_yy, cell2dof.flatten(), stress_yy.flatten())
nodal_stress_yy = nodal_stress_yy / num_count
nodal_stress_zz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_zz, cell2dof.flatten(), stress_zz.flatten())
nodal_stress_zz = nodal_stress_zz / num_count
nodal_stress_xy = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_xy, cell2dof.flatten(), stress_xy.flatten())
nodal_stress_xy = nodal_stress_xy / num_count
nodal_stress_yz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_yz, cell2dof.flatten(), stress_yz.flatten())
nodal_stress_yz = nodal_stress_yz / num_count
nodal_stress_xz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_xz, cell2dof.flatten(), stress_xz.flatten())
nodal_stress_xz = nodal_stress_xz / num_count

# 等效应变
equiv_strain = compute_equivalent_strain(strain, nu) # (NC, 8)
# 等效应力
equiv_stress = compute_equivalent_stress(stress, nu) # (NC, 8)

# 节点等效应变
nodal_equiv_strain = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_equiv_strain, cell2dof.flatten(), equiv_strain.flatten())
nodal_equiv_strain = nodal_equiv_strain / num_count

# 节点等效应力
nodal_equiv_stress = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_equiv_stress, cell2dof.flatten(), equiv_stress.flatten())
nodal_equiv_stress = nodal_equiv_stress / num_count

error_es = nodal_equiv_stress - nodal_equiv_stress_ansys # (NN, )
relative_error_es = bm.linalg.norm(error_es) / (bm.linalg.norm(nodal_equiv_stress)+bm.linalg.norm(nodal_equiv_stress_ansys))
print(f"Relative error_es: {relative_error_es:.12e}")

mesh.nodedata['nodal_equiv_strain'] = nodal_equiv_strain[:]
mesh.nodedata['nodal_equiv_stress'] = nodal_equiv_stress[:]

mesh.nodedata['nodal_strain_xx'] = nodal_strain_xx[:]
mesh.nodedata['nodal_strain_yy'] = nodal_strain_yy[:]
mesh.nodedata['nodal_strain_zz'] = nodal_strain_zz[:]
mesh.nodedata['nodal_strain_xy'] = nodal_strain_xy[:]
mesh.nodedata['nodal_strain_yz'] = nodal_strain_yz[:]
mesh.nodedata['nodal_strain_xz'] = nodal_strain_xz[:]

mesh.nodedata['nodal_stress_xx'] = nodal_stress_xx[:]
mesh.nodedata['nodal_stress_yy'] = nodal_stress_yy[:]
mesh.nodedata['nodal_stress_zz'] = nodal_stress_zz[:]
mesh.nodedata['nodal_stress_xy'] = nodal_stress_xy[:]
mesh.nodedata['nodal_stress_yz'] = nodal_stress_yz[:]
mesh.nodedata['nodal_stress_xz'] = nodal_stress_xz[:]
mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gear15_fealpy.vtu')
print("-----------")