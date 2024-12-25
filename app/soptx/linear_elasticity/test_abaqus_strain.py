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

def compute_strain_stress_at_quadpoints1(space, uh, mu, lam):
    """在积分点处计算应变和应力"""
    mesh = space.mesh
    cell = mesh.entity('cell')
    cell2dof = space.cell_to_dof()
    NC = mesh.number_of_cells()
    NN = mesh.number_of_nodes()

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
        strain[:, i, 3] = bm.sum(
                cuh[:, :, 2]*gphix_q2[:, i, :, 1] + cuh[:, :, 1]*gphix_q2[:, i, :, 2], 
                axis=-1) / 2.0  # (NC,)

        strain[:, i, 4] = bm.sum(
                cuh[:, :, 2]*gphix_q2[:, i, :, 0] + cuh[:, :, 0]*gphix_q2[:, i, :, 2], 
                axis=-1) / 2.0  # (NC,)

        strain[:, i, 5] = bm.sum(
                cuh[:, :, 1]*gphix_q2[:, i, :, 0] + cuh[:, :, 0]*gphix_q2[:, i, :, 1], 
                axis=-1) / 2.0  # (NC,)

    # 计算应力
    val = 2*mu + lam
    stress = bm.zeros((NC, NQ, 6), dtype=bm.float64)
    stress[:, :, 0] = val * strain[:, :, 0] + lam * (strain[:, :, 1] + strain[:, :, 2])
    stress[:, :, 1] = val * strain[:, :, 1] + lam * (strain[:, :, 2] + strain[:, :, 0])
    stress[:, :, 2] = val * strain[:, :, 2] + lam * (strain[:, :, 0] + strain[:, :, 1])
    stress[:, :, 3] = 2*mu * strain[:, :, 3]
    stress[:, :, 4] = 2*mu * strain[:, :, 4]
    stress[:, :, 5] = 2*mu * strain[:, :, 5]

    # 初始化节点累加器和计数器
    nstrain = bm.zeros((NN, 6), dtype=bm.float64)
    nstress = bm.zeros((NN, 6), dtype=bm.float64)

    map = bm.array([0, 4, 6, 2, 1, 5, 7, 3], dtype=bm.int32)
    strain = strain[:, map, :] # (NC, 8, 6)
    stress = stress[:, map, :] # (NC, 8, 6)

    nc = bm.zeros(NN, dtype=bm.int32)
    bm.add_at(nc, cell, 1)
    for i in range(6):
        bm.add_at(nstrain[:, i], cell.flatten(), strain[:, :, i].flatten())
        nstrain[:, i] /= nc
        bm.add_at(nstress[:, i], cell.flatten(), stress[:, :, i].flatten())
        nstress[:, i] /= nc

    return strain, stress, nstrain, nstress

def compute_strain_stress_at_quadpoints3(space, uh, B_BBar, D):
    cell2tdof = space.cell_to_dof()
    cuh = uh[cell2tdof]  # (NC, TLDOF) 
    strain5 = bm.einsum('cqil, cl -> cqi', B_BBar, cuh) # (NC, NQ, 6)
    stress5 = bm.einsum('cqij, cqi -> cqj', D, strain5) # (NC, NQ, 6)

    # 初始化节点累加器和计数器
    mesh = space.mesh
    NN = mesh.number_of_nodes()
    nstrain5 = bm.zeros((NN, 6), dtype=bm.float64)
    nstress5 = bm.zeros((NN, 6), dtype=bm.float64)

    map = bm.array([0, 4, 6, 2, 1, 5, 7, 3], dtype=bm.int32)
    strain5 = strain5[:, map, :] # (NC, 8, 6)
    stress5 = stress5[:, map, :] # (NC, 8, 6)
    nc = bm.zeros(NN, dtype=bm.int32)
    bm.add_at(nc, cell, 1)
    for i in range(6):
        bm.add_at(nstrain5[:, i], cell.flatten(), strain5[:, :, i].flatten())
        nstrain5[:, i] /= nc
        bm.add_at(nstress5[:, i], cell.flatten(), stress5[:, :, i].flatten())
        nstress5[:, i] /= nc
    
    return strain5, stress5, nstrain5, nstress5

node = bm.tensor([[52.938, 11.5135, 33.75],
                  [52.8192, 12.0469, 33.75],
                  [52.3367, 12.2121, 33.75],
                  [52.4869, 11.6464, 33.75],
                  [52.7929, 12.1617, 36],
                  [52.6676, 12.6935, 36],
                  [52.1831, 12.5828, 36],
                  [52.3402, 12.289, 36]], dtype=bm.float64)
cell = bm.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=bm.int32)

mesh = HexahedronMesh(node, cell)
p = 1
q = p+1
space = LagrangeFESpace(mesh, p=p, ctype='C')
tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority

uh_NG = bm.tensor([[0.030791, -0.511217, 0.186367],
                [0.054626, -0.55366, 0.189573],
                [0.0950056, -0.501274, 0.175273],
                [0.029425, -0.548231, 0.193468],
                [0.0852867, -0.404587, 0.148258],
                [0.102669, -0.367209, 0.101832],
                [0.0849172, -0.362382, 0.106393],
                [0.0872416, -0.353984, 0.139331]], dtype=bm.float64)
uh = tensor_space.function()
if tensor_space.dof_priority:
    uh[:] = uh_NG.T.flatten()
else:
    uh[:] = uh_NG.flatten()

strain_abaqus = bm.tensor([[[0.0205907, -0.0373583, 0.00193449, -0.0920545, -0.0178993, 0.0376472],
                           [-0.00639374, -0.0432677, 0.0466738, -0.119936, 0.0206362, 0.0589461],
                           [-0.0352205, 0.0485463, -0.0154576, 0.0108893, -0.0154634, -0.0329297],
                           [-0.00058211, -0.0206767, 0.0069492, -0.00977775, 0.00676531, 0.0219284],
                           [-0.00837886, 0.0178503, -0.0165338, -0.0490841, 0.0105336, -0.0368539],
                           [0.0127293, 0.0338702, -0.0530039, 0.0597673, -0.0297061, -0.0337809],
                           [0.0246274, -0.0171114, -0.0124114, -0.00879073, -0.0415487, -0.00253086],
                           [0.00543309, 0.0117187, -0.0250226, 0.00239844, 0.00708347, 0.0160583]]], dtype=bm.float64)

E = 206e3
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K = LinearElasticIntegrator(material=linear_elastic_material, 
                                       q=q, method='C3D8_BBar')
integrator_K.keep_data(True)
_, _, D, B_BBar = integrator_K.fetch_c3d8_bbar_assembly(tensor_space)

strain3, stress3, nstrain3, nstress3 = compute_strain_stress_at_quadpoints1(
                                            space, uh_NG, mu, lam)

strain5, stress5, nstrain5, nstress5 = compute_strain_stress_at_quadpoints3(
                                            tensor_space, uh, B_BBar, D
                                        )
error_strain_3 = bm.linalg.norm(strain_abaqus - strain3, axis=1)
error_strain_5 = bm.linalg.norm(strain_abaqus - strain5, axis=1)
p = 1
q = p+1
space = LagrangeFESpace(mesh, p=p, ctype='C')
sgdof = space.number_of_global_dofs()
print(f"sgdof: {sgdof}")
tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K_sri = LinearElasticIntegrator(material=linear_elastic_material,
                                           q=q, method='C3D8_SRI')
qf2 = mesh.quadrature_formula(q)
bcs2, ws2 = qf2.get_quadrature_points_and_weights()
gphi2 = space.grad_basis(bcs2, variable='x')

# B0_q1 = linear_elastic_material._normal_strain_sri(gphi=gphi1)
KE_sri_yz_xz_xy = integrator_K_sri.c3d8_sri_assembly(space=tensor_space)
print("---------------------")