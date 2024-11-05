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


with open('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/external_gear_test_data.pkl', 'rb') as f:
    data = pickle.load(f)
hex_mesh = data['hex_mesh']
helix_node = data['helix_node']
inner_node_idx = data['inner_node_idx']


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
cell_target = cell[target_cell_idx, :]

load_values = bm.array([50.0, 60.0, 79.0, 78.0, 87.0, 95.0, 102.0, 109.0, 114.0,
                        119.0, 123.0, 127.0, 129.0, 130.0, 131.0], dtype=bm.float64)

u = parameters[..., 0]
v = parameters[..., 1]
w = parameters[..., 2]
bcs_list = [
    (
        bm.tensor([[u, 1 - u]]),
        bm.tensor([[v, 1 - v]]),
        bm.tensor([[w, 1 - w]])
    )
    for u, v, w in parameters
]

space = LagrangeFESpace(mesh, p=1, ctype='C')
scalar_gdof = space.number_of_global_dofs()
tensor_space = TensorFunctionSpace(space, shape=(3, -1))
tgdof = tensor_space.number_of_global_dofs()
tldof = tensor_space.number_of_local_dofs()
cell2tdof = tensor_space.cell_to_dof()

scalar_is_bd_dof = is_inner_node
tensor_is_bd_dof = tensor_space.is_boundary_dof(
        threshold=(scalar_is_bd_dof, scalar_is_bd_dof, scalar_is_bd_dof), 
        method='interp')

phi_loads = []
for bcs in bcs_list:
    phi = tensor_space.basis(bcs)
    phi_loads.append(phi)

phi_loads_array = bm.concatenate(phi_loads, axis=1) # (1, NP, tldof, GD)

FE_load = bm.einsum('p, cpld -> pl', load_values, phi_loads_array) # (NP, tldof)
FE = bm.zeros((NC, tldof), dtype=bm.float64)
FE[target_cell_idx, :] = FE_load[:, :] # (NC, tldof)

F = COOTensor(indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, ))
indices = cell2tdof.reshape(1, -1)
F = F.add(COOTensor(indices, FE.reshape(-1), (tgdof, ))).to_dense() # (tgdof, )

linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='3D', device=bm.get_device(mesh))

integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=2)
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')
# K_dense = K.to_dense()

def dirichlet(points: TensorLike) -> TensorLike:
    return bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))

dbc = DirichletBC(space=tensor_space, 
                    gd=dirichlet, 
                    threshold=tensor_is_bd_dof, 
                    method='interp')
K, F = dbc.apply(A=K, f=F, check=True)

uh = tensor_space.function()

uh[:] = cg(K, F, maxiter=100, atol=1e-12, rtol=1e-12)
uh = uh.reshape(GD, NN).T

mesh.nodedata['deform'] = uh[:]
mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gearx_mesh.vtu')
print("-----------")