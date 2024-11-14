"""
一个载荷点，不处理重心坐标，考虑外法线方向的载荷的验证
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


with open('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/external_gear_data_part.pkl', 'rb') as f:
    data = pickle.load(f)

hex_mesh = data['hex_mesh']
helix_node = data['helix_node']
target_cells_idx = data['target_cell_idx']
parameters = data['parameters']
is_inner_node = data['is_inner_node']

parameter = parameters[-1:] # (1, 3)
target_cell_idx = target_cells_idx[-1:] # (1, )

hex_cell = hex_mesh.cell
hex_node = hex_mesh.node

mesh = HexahedronMesh(hex_node, hex_cell)

GD = mesh.geo_dimension()   
NC = mesh.number_of_cells()
NN = mesh.number_of_nodes()
node = mesh.entity('node')
cell = mesh.entity('cell')

# 带有载荷的节点的全局编号
load_node_indices = cell[target_cell_idx].flatten() # (1*8, )

# 点载荷值
load_values = bm.array([131.0], dtype=bm.float64)
# 单位外法向量 
cellnorm = mesh.cell_normal() # (NC, 3)

# 点载荷向量
target_cellnorm = cellnorm[target_cell_idx] # (3, )
P = bm.einsum('p, pd -> pd', load_values, target_cellnorm)  # (1, 3)

u = parameter[..., 0]
v = parameter[..., 1]
w = parameter[..., 2]

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
tensor_space = TensorFunctionSpace(space, shape=(3, -1))
tgdof = tensor_space.number_of_global_dofs()
tldof = tensor_space.number_of_local_dofs()
cell2tdof = tensor_space.cell_to_dof()

# 带有载荷的节点对应的全局自由度编号
dof_indices = bm.stack([scalar_gdof * d + load_node_indices for d in range(GD)], axis=1)  # (1*8, 3)

phi_loads = []
for bcs in bcs_list:
    phi = tensor_space.basis(bcs)
    phi_loads.append(phi)

phi_loads_array = bm.concatenate(phi_loads, axis=1) # (1, NP, tldof, GD)

FE_load = bm.einsum('pd, cpld -> pl', P, phi_loads_array) # (1, 24)

FE = bm.zeros((NC, tldof), dtype=bm.float64)
FE[target_cell_idx, :] = FE_load[:, :] # (NC, tldof)

F = COOTensor(indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, ))
indices = cell2tdof.reshape(1, -1)
F = F.add(COOTensor(indices, FE.reshape(-1), (tgdof, ))).to_dense() # (tgdof, )

# 从全局载荷向量中提取有载荷节点处的值
F_load_nodes = F[dof_indices] # (1*8, 3)
print("----------------------------")
