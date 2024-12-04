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


with open('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/external_gear_data_part.pkl', 'rb') as f:
    data = pickle.load(f)

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
NN = mesh.number_of_nodes()
node = mesh.entity('node')
cell = mesh.entity('cell')

load_values = bm.array([50.0, 60.0, 79.0, 78.0, 87.0, 95.0, 102.0, 109.0, 114.0,
                        119.0, 123.0, 127.0, 129.0, 130.0, 131.0], dtype=bm.float64)

cellnorm = mesh.cell_normal() # (NC, GD)

target_cellnorm = cellnorm[target_cell_idx] # (15, GD)
P = bm.einsum('p, pd -> pd', load_values, target_cellnorm)  # (15, GD)

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

q = 2

# tensor_space = TensorFunctionSpace(space, shape=(-1, 3)) # gd_priority
tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority

tgdof = tensor_space.number_of_global_dofs()
tldof = tensor_space.number_of_local_dofs()
cell2tdof = tensor_space.cell_to_dof()

phi_loads = []
for bcs in bcs_list:
    phi = tensor_space.basis(bcs)
    phi_loads.append(phi)

phi_loads_array = bm.concatenate(phi_loads, axis=1) # (1, NP, tldof, GD)

FE_load = bm.einsum('pd, cpld -> pl', P, phi_loads_array) # (NP, 24)

FE = bm.zeros((NC, tldof), dtype=bm.float64)
FE[target_cell_idx, :] = FE_load[:, :] # (NC, tldof)

F = COOTensor(indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, ))
indices = cell2tdof.reshape(1, -1)
F = F.add(COOTensor(indices, FE.reshape(-1), (tgdof, ))).to_dense() # (tgdof, )

E = 206e3
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))

integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=q)

KE = integrator_K.assembly(space=tensor_space)
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')

from app.gearx.utils import *
F_load_nodes = bm.transpose(F.reshape(3, -1))

@cartesian
def dirichlet(points: TensorLike) -> TensorLike:
    result = bm.zeros(points.shape, 
                        dtype=points.dtype, device=bm.get_device(points))

    return result

scalar_is_bd_dof = is_inner_node
scalar_num = bm.sum(scalar_is_bd_dof)
tensor_is_bd_dof = tensor_space.is_boundary_dof(
        threshold=(scalar_is_bd_dof, scalar_is_bd_dof, scalar_is_bd_dof), 
        method='interp')

dbc = DirichletBC(space=tensor_space, 
                    gd=dirichlet, 
                    threshold=tensor_is_bd_dof, 
                    method='interp')

K = dbc.apply_matrix(matrix=K, check=True)

uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), dtype=bm.float64, device=bm.get_device(mesh))
uh_bd, isDDof = tensor_space.boundary_interpolate(gd=dirichlet, 
                                                uh=uh_bd, 
                                                threshold=tensor_is_bd_dof, 
                                                method='interp')

F = F - K.matmul(uh_bd)
F = bm.set_at(F, isDDof, uh_bd[isDDof])

from fealpy import logger
logger.setLevel('INFO')

uh = tensor_space.function()
uh[:] = cg(K, F, maxiter=10000, atol=1e-8, rtol=1e-8)
# uh[:] = spsolve(K, F, solver='mumps')

# 计算残差向量和范数
residual = K.matmul(uh[:]) - F  # 使用 CSRTensor 的 matmul 方法
residual_norm = bm.sqrt(bm.sum(residual * residual))
print(f"Final residual norm: {residual_norm:.6e}")

uh = uh.reshape(GD, NN).T

mesh.nodedata['deform'] = uh[:]
mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gear.vtu')
print("-----------")