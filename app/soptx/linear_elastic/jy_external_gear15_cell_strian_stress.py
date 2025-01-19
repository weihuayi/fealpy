"""
外齿轮 15 个载荷点的算例
"""
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

def compute_strain_stress_1(space, uh, mu, lam):
    """在积分点处计算应变和应力"""
    mesh = space.mesh
    cell = mesh.entity('cell')
    cell2dof = space.cell_to_dof()
    NC = mesh.number_of_cells()

    # 计算积分点处的基函数梯度
    qf2 = mesh.quadrature_formula(2)
    bcs_q2, ws = qf2.get_quadrature_points_and_weights()
    gphix_q2 = space.grad_basis(bcs_q2, variable='x')  # (NC, NQ, LDOF, GD)

    cuh = uh[cell2dof]  # (NC, LDOF, GD)

    # 计算应变
    NQ = len(ws)  # 积分点个数
    strain = bm.zeros((NC, NQ, 6), dtype=bm.float64)
    
    # 计算正应变和剪切应变
    strain[:, :, 0:3] = bm.einsum('cid, cqid -> cqd', cuh, gphix_q2)  # (NC, NQ, 3)
    for i in range(NQ):  
        # strain[:, i, 3] = bm.sum(
        #         cuh[:, :, 2]*gphix_q2[:, i, :, 1] + cuh[:, :, 1]*gphix_q2[:, i, :, 2], 
        #         axis=-1) / 2.0  # (NC,)

        # strain[:, i, 4] = bm.sum(
        #         cuh[:, :, 2]*gphix_q2[:, i, :, 0] + cuh[:, :, 0]*gphix_q2[:, i, :, 2], 
        #         axis=-1) / 2.0  # (NC,)

        # strain[:, i, 5] = bm.sum(
        #         cuh[:, :, 1]*gphix_q2[:, i, :, 0] + cuh[:, :, 0]*gphix_q2[:, i, :, 1], 
        #         axis=-1) / 2.0  # (NC,)
        strain[:, i, 3] = bm.sum(
                cuh[..., 1]*gphix_q2[:, i, :, 0] + cuh[..., 0]*gphix_q2[:, i, :, 1], axis=-1)  # (NC,)
        strain[:, i, 4] = bm.sum(
                cuh[..., 2]*gphix_q2[:, i, :, 1] + cuh[..., 1]*gphix_q2[:, i, :, 2], axis=-1)  # (NC,)
        strain[:, i, 5] = bm.sum(
                cuh[..., 2]*gphix_q2[:, i, :, 0] + cuh[..., 0]*gphix_q2[:, i, :, 2], axis=-1)  # (NC,)

    # 计算应力
    val = 2*mu + lam
    stress = bm.zeros((NC, NQ, 6), dtype=bm.float64)
    stress[:, :, 0] = val * strain[:, :, 0] + lam * (strain[:, :, 1] + strain[:, :, 2])
    stress[:, :, 1] = val * strain[:, :, 1] + lam * (strain[:, :, 2] + strain[:, :, 0])
    stress[:, :, 2] = val * strain[:, :, 2] + lam * (strain[:, :, 0] + strain[:, :, 1])
    # stress[:, :, 3] = 2*mu * strain[:, :, 3]
    # stress[:, :, 4] = 2*mu * strain[:, :, 4]
    # stress[:, :, 5] = 2*mu * strain[:, :, 5]
    stress[:, :, 3] = mu * strain[:, :, 3]
    stress[:, :, 4] = mu * strain[:, :, 4]
    stress[:, :, 5] = mu * strain[:, :, 5]

    return strain, stress

def compute_strain_stress_2(tensor_space, uh, B, D):
    cell2tdof = tensor_space.cell_to_dof()
    cuh = uh[cell2tdof]  # (NC, TLDOF) 
    strain = bm.einsum('cqil, cl -> cqi', B, cuh) # (NC, NQ, 6)
    stress = bm.einsum('cqij, cqi -> cqj', D, strain) # (NC, NQ, 6)
    
    return strain, stress

def compute_strain_stress_3(tensor_space, uh, B_BBar, D):
    cell2tdof = tensor_space.cell_to_dof()
    cuh = uh[cell2tdof]  # (NC, TLDOF) 
    strain = bm.einsum('cqil, cl -> cqi', B_BBar, cuh) # (NC, NQ, 6)
    stress = bm.einsum('cqij, cqi -> cqj', D, strain) # (NC, NQ, 6)
    
    return strain, stress

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

# # Ansys 位移结果

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
                        119.0, 123.0, 127.0, 129.0, 130.0, 131.0], dtype=bm.float64)   # (15, )

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
q = 2
space = LagrangeFESpace(mesh, p=p, ctype='C')
scalar_gdof = space.number_of_global_dofs()
print(f"gdof: {scalar_gdof}")
cell2dof = space.cell_to_dof()
tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority

tgdof = tensor_space.number_of_global_dofs()
print(f"tgdof: {tgdof}")
tldof = tensor_space.number_of_local_dofs()
cell2tdof = tensor_space.cell_to_dof()

# 节点载荷的索引
load_node_indices0 = cell[target_cell_idx].flatten() # (15*8, )
unique_nodes, first_indices = bm.unique(load_node_indices0, return_index=True)
sort1_indices = bm.sort(first_indices)
load_node_indices = load_node_indices0[sort1_indices] # (15*8, )
# 带有载荷的节点对应的全局自由度编号（跟顺序有关）
if tensor_space.dof_priority:
    dof_indices = bm.stack([scalar_gdof * d + load_node_indices for d in range(GD)], axis=1)  # (15*8, GD)
else:
    dof_indices = bm.stack([load_node_indices * GD + d for d in range(GD)], axis=1)  # (15*8, GD)

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
export_to_inp(filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/inp/external_gear_abaqus.inp', 
              nodes=node, elements=cell, 
              fixed_nodes=fixed_node_index, load_nodes=load_node_indices, loads=F_load_nodes, 
              young_modulus=206e3, poisson_ratio=0.3, density=7.85e-9, 
              used_app='abaqus', mesh_type='hex')

E = 206e3
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K0 = LinearElasticIntegrator(material=linear_elastic_material, 
                                        q=q, method='voigt')
integrator_K0.keep_data(True)
_, _, _, D, B = integrator_K0.fetch_voigt_assembly(tensor_space)
integrator_K = LinearElasticIntegrator(material=linear_elastic_material, 
                                       q=q, method='C3D8_BBar')
integrator_K.keep_data(True)
_, _, D, B_BBar = integrator_K.fetch_c3d8_bbar_assembly(tensor_space)
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')

# 处理 Dirichlet 边界条件
scalar_is_bd_dof = bm.zeros(scalar_gdof, dtype=bm.bool)
scalar_is_bd_dof[:NN] = is_inner_node
tensor_is_bd_dof = tensor_space.is_boundary_dof(
                                threshold=(scalar_is_bd_dof, scalar_is_bd_dof, scalar_is_bd_dof), 
                                method='interp')
dbc = DirichletBC(space=tensor_space, 
                    gd=bm.zeros(tgdof), 
                    threshold=tensor_is_bd_dof, 
                    method='interp')
K, F = dbc.apply(K, F)

from fealpy import logger
logger.setLevel('INFO')
uh = tensor_space.function()
# uh[:] = cg(K, F, maxiter=10000, atol=1e-8, rtol=1e-8)
uh[:] = spsolve(K, F, solver="mumps")

# 计算残差向量和范数
residual = K.matmul(uh[:]) - F  # 使用 CSRTensor 的 matmul 方法
residual_norm = bm.sqrt(bm.sum(residual * residual))
print(f"Final residual norm: {residual_norm:.6e}")

if tensor_space.dof_priority:
    uh_show = uh.reshape(GD, NN).T
else:
    uh_show = uh.reshape(NN, GD)
cuh = uh_show[cell] # (NC, NCN, GD)
map_cuh = [0, 1, 3, 2, 4, 5, 7, 6]
print(f"cuh_0-{cuh[0].shape}, {cell[0]}\n {cuh[0]}")
print(f"cuh_0_map-{cuh[0].shape}, {cell[0][map_cuh]}\n {cuh[0][map_cuh, :]}")
# print(f"cuh_1-{cuh[1].shape}, {cell[1]}\n {cuh[1]}")

uh_magnitude = bm.linalg.norm(uh_show, axis=1)

mesh.nodedata['uh'] = uh_show[:]
mesh.nodedata['uh_magnitude'] = uh_magnitude[:]

cuh_0_ansys = bm.tensor([[0.0000,     0.0000,     0.000],
                        [-6.4276e-4, -1.3507e-3, -4.7196e-4],
                        [0.0000,     0.0000,     0.000],
                        [-6.0845e-4, -1.3320e-3, -4.4468e-4],
                        [0.0000,     0.0000,     0.000],
                        [-5.0779e-4, -1.5781e-3, -2.2451e-4],
                        [0.0000,     0.0000,     0.000],
                        [-4.8505e-4, -1.5615e-3, -2.1318e-4]], dtype=bm.float64)
print(f"cuh_0_ansys-{cuh_0_ansys.shape}, {[0, 24, 331, 364, 2178, 2202, 2509, 2542]}\n {cuh_0_ansys}")
error_l2_cuh_0 = bm.linalg.norm(cuh[0][map_cuh, :] - cuh_0_ansys)
print(f"error_l2_cuh_0: {error_l2_cuh_0:.8e}")
error_l2_rel_cuh_0 = bm.linalg.norm(cuh[0][map_cuh, :] - cuh_0_ansys) / (bm.linalg.norm(cuh_0_ansys))
print(f"error_l2_rel_cuh_0: {error_l2_rel_cuh_0:.8e}")
error_max_cuh_0 = bm.max(bm.abs(cuh[0][map_cuh, :] - cuh_0_ansys))
print(f"error_max_cuh_0: {error_max_cuh_0:.8e}")

np.set_printoptions(precision=4, suppress=False)

# NOTE 应变
strain1, stress1 = compute_strain_stress_1(space, uh_show, mu, lam)
map_es = [0, 4, 2, 6, 1, 5, 3, 7]
print(f"strain1_0-{strain1[0].shape}, {cell2dof[0]}\n {strain1[0]}")
print(f"strain1_0_map-{strain1[0].shape}, {cell2dof[0][map_es]}\n {strain1[0][map_es, :]}")

strain2, stress2 = compute_strain_stress_2(tensor_space, uh, B, D)

strain3, stress3 = compute_strain_stress_3(tensor_space, uh, B_BBar, D)
print(f"strain_BBar_0: {strain3.shape}, {cell2dof[0]}\n {strain3[0]}")
print(f"strain_BBar_0_map: {strain3.shape}, {cell2dof[0][map_es]}\n {strain3[0][map_es, :]}")

error12 = bm.linalg.norm(strain1 - strain2)
error13 = bm.linalg.norm(strain1 - strain3)
print(f"error12: {error12:.12e}")
print(f"error13: {error13:.12e}")

# [xx, yy, zz, xy, yz, xz]
strain_0_ansys = bm.tensor([
                    [-6.4997e-4, 2.6777e-4, 4.0435e-5, -1.3961e-3, 4.7005e-5, -3.6421e-4],
                    [-6.9321e-4, 3.2490e-4, 9.2012e-5, -1.3546e-3, 4.8227e-5, -3.6566e-4],
                    [-6.3136e-4, 2.6192e-4, 3.8964e-5, -1.4010e-3, 4.5833e-5, -3.4739e-4],
                    [-6.7491e-4, 3.1914e-4, 8.8178e-5, -1.3595e-3, 4.6737e-5, -3.5085e-4],
                    [-5.6564e-4, 2.7827e-4, -4.3553e-7, -1.5713e-3, 5.8720e-5, -3.6197e-4],
                    [-6.1220e-4, 3.3237e-4, 5.5974e-5, -1.5480e-3, 3.5611e-5, -3.6204e-4],
                    [-5.5134e-4, 2.7262e-4, -1.8154e-5, -1.5761e-3, 5.8458e-5, -3.5106e-4],
                    [-5.9813e-4, 3.2681e-4, 5.2232e-5, -1.5528e-3, 3.5012e-5, -3.5305e-4]
                ], dtype=bm.float64)
print(f"strain_0_ansys: {strain_0_ansys.shape}, {[0, 24, 331, 364, 2178, 2202, 2509, 2542]}\n {strain_0_ansys}")

# NOTE 应变误差
error_l2_strain_0 = bm.linalg.norm(strain1[0][map_es, :] - strain_0_ansys)
print(f"error_l2_strain_0: {error_l2_strain_0:.8e}")
error_l2_rel_strain_0 = bm.linalg.norm(strain1[0][map_es, :] - strain_0_ansys) / bm.linalg.norm(strain_0_ansys)
print(f"error_l2_rel_strain_0: {error_l2_rel_strain_0:.8e}")
error_max_strain_0 = bm.max(bm.abs(strain1[0][map_es, :] - strain_0_ansys))
print(f"error_max_strain_0: {error_max_strain_0:.8e}")

error_l2_strain_0_BBar = bm.linalg.norm(strain3[0][map_es, :] - strain_0_ansys)
print(f"error_l2_strain_0_BBar: {error_l2_strain_0_BBar:.8e}")
error_l2_rel_strain_0_BBar = bm.linalg.norm(strain3[0][map_es, :] - strain_0_ansys) / bm.linalg.norm(strain_0_ansys)
print(f"error_l2_rel_strain_0_BBar: {error_l2_rel_strain_0_BBar:.8e}")
error_max_strain_0_BBar = bm.max(bm.abs(strain3[0][map_es, :] - strain_0_ansys)) 
print(f"error_max_strain_0_BBar: {error_max_strain_0_BBar:.8e}")


#NOTE 应力
print(f"stress1_0: {stress1[0].shape}, {cell2dof[0]}\n {stress1[0]}")
print(f"stress1_0_map: {stress1[0].shape}, {cell2dof[0][map_es]}\n {stress1[0][map_es, :]}")
print(f"stress_BBar_0: {stress3.shape}, {cell2dof[0]}\n {stress3[0]}")
print(f"stress_BBar_0_map: {stress3.shape}, {cell2dof[0][map_es]}\n {stress3[0][map_es, :]}")

# [xx, yy, zz, xy, yz, xz]
stress_0_ansys = bm.tensor([
                        [-143.61,   1.8134, -34.210, -110.62,   3.7242, -28.857],
                        [-142.68,  18.6460, -18.257, -107.33,   3.8211, -28.971],
                        [-139.32,   2.2279, -33.102, -111.01,   3.6314, -27.524],
                        [-138.75,  18.7700, -17.829, -107.71,   3.7030, -27.798],
                        [-123.84,   9.8896, -34.274, -124.49,   4.6524, -28.679],
                        [-123.61,  26.0630, -17.735, -122.65,   2.8215, -28.685],
                        [-120.71,   9.8589, -33.628, -124.88,   4.6317, -27.814],
                        [-120.82,  25.7490, -17.761, -123.03,   2.7740, -27.972]
                    ], dtype=bm.float64)
print(f"stress_0_ansys: {stress_0_ansys.shape}, {[0, 24, 331, 364, 2178, 2202, 2509, 2542]}\n {stress_0_ansys}")

# NOTE 应力误差
error_l2_stress_0 = bm.linalg.norm(stress1[0][map_es, :] - stress_0_ansys)
print(f"error_l2_stress_0: {error_l2_stress_0:.8e}")
error_l2_rel_stress_0 = bm.linalg.norm(stress1[0][map_es, :] - stress_0_ansys) / bm.linalg.norm(stress_0_ansys)
print(f"error_l2_rel_stress_0: {error_l2_rel_stress_0:.8e}")
error_max_stress_0 = bm.max(bm.abs(stress1[0][map_es, :] - stress_0_ansys))
print(f"error_max_stress_0: {error_max_stress_0:.8e}")

error_l2_stress_0_BBar = bm.linalg.norm(stress3[0][map_es, :] - stress_0_ansys)
print(f"error_l2_stress_0_BBar: {error_l2_stress_0_BBar:.8e}")
error_l2_rel_stress_0_BBar = bm.linalg.norm(stress3[0][map_es, :] - stress_0_ansys) / bm.linalg.norm(stress_0_ansys)
print(f"error_l2_rel_stress_0_BBar: {error_l2_rel_stress_0_BBar:.8e}")
error_max_stress_0_BBar = bm.max(bm.abs(stress3[0][map_es, :] - stress_0_ansys))
print(f"error_max_stress_0_BBar: {error_max_stress_0_BBar:.8e}")

mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/vtu/external_gear_fealpy.vtu')
print("-----------")