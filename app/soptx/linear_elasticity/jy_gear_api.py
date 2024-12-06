from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.sparse import COOTensor
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.typing import TensorLike
from fealpy.solver import cg, spsolve
from fealpy.decorator import cartesian

import pickle
from app.gearx.gear import ExternalGear, InternalGear
from app.gearx.utils import *

from fealpy.mesh import HexahedronMesh

def compute_equivalent_strain(strain, nu):
    # Extract strain components
    exx = strain[..., 0, 0]
    eyy = strain[..., 1, 1]
    ezz = strain[..., 2, 2]
    gamma_xy = strain[..., 0, 1]
    gamma_yz = strain[..., 1, 2]
    gamma_xz = strain[..., 0, 2]
    
    # Normal strain differences
    d1 = exx - eyy
    d2 = eyy - ezz
    d3 = ezz - exx
    
    # Combine all terms
    equiv_strain = (d1**2 + d2**2 + d3**2 + 6.0 * (gamma_xy**2 + gamma_yz**2 + gamma_xz**2))
    
    # Final computation with Poisson's ratio factor and square root
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
    
    # Combine all terms
    equiv_stress = (d1**2 + d2**2 + d3**2 + 6.0 * (sxy**2 + syz**2 + sxz**2))

    equiv_stress = bm.sqrt(equiv_stress / 2.0)
    
    return equiv_stress

with open('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/external_gear_data_part.pkl', 'rb') as f:
    data = pickle.load(f)

external_gear = data['external_gear']
hex_mesh = data['hex_mesh']
helix_node = data['helix_node']
target_cell_idx = data['target_cell_idx']
parameters = data['parameters']
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

load_values = bm.array([50.0, 60.0, 79.0, 78.0, 87.0, 95.0, 102.0, 109.0, 114.0,
                        119.0, 123.0, 127.0, 129.0, 130.0, 131.0], dtype=bm.float64)

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

space = LagrangeFESpace(mesh, p=1, ctype='C')
scalar_gdof = space.number_of_global_dofs()
cell2dof = space.cell_to_dof()

q = 2

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

# KE = integrator_K.assembly(space=tensor_space)
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"Matrix norm after dc: {K_norm:.6f}")
print(f"Load vector norm after dc: {F_norm:.6f}")


scalar_is_bd_dof = is_inner_node
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
uh[:] = cg(K, F, maxiter=10000, atol=1e-8, rtol=1e-8)

# 计算残差向量和范数
residual = K.matmul(uh[:]) - F  # 使用 CSRTensor 的 matmul 方法
residual_norm = bm.sqrt(bm.sum(residual * residual))
print(f"Final residual norm: {residual_norm:.6e}")

if tensor_space.dof_priority:
    uh_show = uh.reshape(GD, NN).T
else:
    uh_show = uh.reshape(NN, GD)

uh_magnitude = bm.linalg.norm(uh_show, axis=1)

mesh.nodedata['uh'] = uh_show[:]
mesh.nodedata['uh_magnitude'] = uh_magnitude[:]

uh_cell = bm.zeros((NC, tldof))
for c in range(NC):
    uh_cell[c] = uh[cell2tdof[c]]

# bcs1 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64),)
# bcs5 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64),)
# bcs7 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64),)
# bcs3 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64),)
# bcs2 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64),)
# bcs6 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64),)
# bcs8 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64),)
# bcs4 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64),)

bcs1 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
bcs2 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
bcs3 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
bcs4 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64),)
bcs5 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)
bcs6 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)
bcs7 = (bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)
bcs8 = (bm.tensor([[1, 0]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64), 
       bm.tensor([[0, 1]], dtype=bm.float64),)

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

tgrad = bm.einsum('cqimn, ci -> cqmn', tgphi, uh_cell)      # (NC, 8, GD, GD)
strain = (tgrad + bm.transpose(tgrad, (0, 1, 3, 2))) / 2 # (NC, 8, GD, GD)
strain_xx = strain[..., 0, 0] # (NC, 8)
strain_yy = strain[..., 1, 1] # (NC, 8)
strain_zz = strain[..., 2, 2] # (NC, 8)
strain_xy = strain[..., 0, 1] # (NC, 8)
strain_yz = strain[..., 1, 2] # (NC, 8)
strain_xz = strain[..., 0, 2] # (NC, 8)

trace_e = bm.einsum("...ii", strain) # (NC, 8)
I = bm.eye(GD, dtype=bm.float64)
stress = 2 * mu * strain + lam * trace_e[..., None, None] * I # (NC, 8, GD, GD)

stress_xx = stress[..., 0, 0] # (NC, 8)
stress_yy = stress[..., 1, 1] # (NC, 8)
stress_zz = stress[..., 2, 2] # (NC, 8)
stress_xy = stress[..., 0, 1] # (NC, 8)
stress_yz = stress[..., 1, 2] # (NC, 8)
stress_xz = stress[..., 0, 2] # (NC, 8)

equiv_strain = compute_equivalent_strain(strain, nu) # (NC, 8)

equiv_stress = compute_equivalent_stress(stress, nu) # (NC, 8)

num_count = bm.zeros(NN, dtype=bm.int32)
bm.add_at(num_count, cell2dof.flatten(), 1)

nodal_equiv_strain = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_equiv_strain, cell2dof.flatten(), equiv_strain.flatten())
nodal_equiv_strain = nodal_equiv_strain / num_count

nodal_equiv_stress = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_equiv_stress, cell2dof.flatten(), equiv_stress.flatten())
nodal_equiv_stress = nodal_equiv_stress / num_count

nodal_strain_xx = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_strain_xx, cell2dof.flatten(), strain_xx.flatten())
nodal_strain_yy = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_strain_yy, cell2dof.flatten(), strain_yy.flatten())
nodal_strain_zz = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_strain_zz, cell2dof.flatten(), strain_zz.flatten())
nodal_strain_xy = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_strain_xy, cell2dof.flatten(), strain_xy.flatten())
nodal_strain_yz = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_strain_yz, cell2dof.flatten(), strain_yz.flatten())
nodal_strain_xz = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_strain_xz, cell2dof.flatten(), strain_xz.flatten())

nodal_stress_xx = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_stress_xx, cell2dof.flatten(), stress_xx.flatten())
nodal_stress_yy = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_stress_yy, cell2dof.flatten(), stress_yy.flatten())
nodal_stress_zz = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_stress_zz, cell2dof.flatten(), stress_zz.flatten())
nodal_stress_xy = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_stress_xy, cell2dof.flatten(), stress_xy.flatten())
nodal_stress_yz = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_stress_yz, cell2dof.flatten(), stress_yz.flatten())
nodal_stress_xz = bm.zeros(NN, dtype=bm.float64)
bm.set_at(nodal_stress_xz, cell2dof.flatten(), stress_xz.flatten())

mesh.celldata['equiv_strain'] = equiv_strain[:]
mesh.celldata['equiv_stress'] = equiv_stress[:]
mesh.celldata['strain_xx'] = strain_xx[:]
mesh.celldata['strain_yy'] = strain_yy[:]
mesh.celldata['strain_zz'] = strain_zz[:]
mesh.celldata['strain_xy'] = strain_xy[:]
mesh.celldata['strain_yz'] = strain_yz[:]
mesh.celldata['strain_xz'] = strain_xz[:]
mesh.celldata['stress_xx'] = stress_xx[:]
mesh.celldata['stress_yy'] = stress_yy[:]
mesh.celldata['stress_zz'] = stress_zz[:]
mesh.celldata['stress_xy'] = stress_xy[:]
mesh.celldata['stress_yz'] = stress_yz[:]
mesh.celldata['stress_xz'] = stress_xz[:]
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
mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gear_fealpy.vtu')

# # 首先将应变张量转换为Voigt记号形式
# strain_voigt = bm.zeros((NC, 8, 6), dtype=bm.float64)
# strain_voigt[..., 0] = strain[..., 0, 0]  # εxx
# strain_voigt[..., 1] = strain[..., 1, 1]  # εyy
# strain_voigt[..., 2] = strain[..., 2, 2]  # εzz
# strain_voigt[..., 3] = 2 * strain[..., 0, 1]  # 2εxy
# strain_voigt[..., 4] = 2 * strain[..., 1, 2]  # 2εyz
# strain_voigt[..., 5] = 2 * strain[..., 0, 2]  # 2εxz

# error_xx = strain_xx - strain_voigt[..., 0]
# error_yy = strain_yy - strain_voigt[..., 1]
# error_zz = strain_zz - strain_voigt[..., 2]
# error_xy = strain_xy - strain_voigt[..., 3] / 2
# error_yz = strain_yz - strain_voigt[..., 4] / 2
# error_xz = strain_xz - strain_voigt[..., 5] / 2

# # 使用弹性矩阵计算应力（Voigt记号形式）
# D = linear_elastic_material.elastic_matrix() # (1, 1, 6, 6)
# D_expanded = bm.broadcast_to(D, (NC, 8, 6, 6))
# stress_voigt = bm.einsum('cqij, cqj -> cqi', D_expanded, strain_voigt)  # (NC, 8, 6)

# # 将 Voigt 记号形式的应力转换回张量形式
# stress = bm.zeros((NC, 8, GD, GD), dtype=bm.float64)
# stress[..., 0, 0] = stress_voigt[..., 0]  # σxx
# stress[..., 1, 1] = stress_voigt[..., 1]  # σyy
# stress[..., 2, 2] = stress_voigt[..., 2]  # σzz
# stress[..., 0, 1] = stress_voigt[..., 3] / 2  # σxy
# stress[..., 1, 0] = stress_voigt[..., 3] / 2  # σyx
# stress[..., 1, 2] = stress_voigt[..., 4] / 2  # σyz
# stress[..., 2, 1] = stress_voigt[..., 4] / 2  # σzy
# stress[..., 0, 2] = stress_voigt[..., 5] / 2  # σxz
# stress[..., 2, 0] = stress_voigt[..., 5] / 2  # σzx
print("-----------")