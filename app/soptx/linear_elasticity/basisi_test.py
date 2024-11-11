from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

import pickle
from app.gearx.gear import ExternalGear, InternalGear
from app.gearx.utils import *

from fealpy.mesh import HexahedronMesh

bm.set_backend('numpy')
nx, ny, nz = 1, 1, 1
mesh_fealpy = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], 
                            nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))
space_fealpy = LagrangeFESpace(mesh_fealpy, p=1, ctype='C')
node_fealpy = mesh_fealpy.entity('node')
cell_fealpy = mesh_fealpy.entity('cell')
cell2dof_fealpy = space_fealpy.cell_to_dof()
# ip = mesh_fealpy.interpolation_points(p=1)
# import matplotlib.pyplot as plt
# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh_fealpy.add_plot(axes)
# mesh_fealpy.find_node(axes, node=ip, showindex=True)
# mesh_fealpy.find_cell(axes, showindex=True)
# plt.show()

q = 2
qf = mesh_fealpy.quadrature_formula(q)
bcs_fealpy, _ = qf.get_quadrature_points_and_weights()
phi_fealpy = space_fealpy.basis(bcs_fealpy) # (1, NQ, ldof)
phi_fealpy_show = phi_fealpy[0]

with open('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/external_gear_data_part.pkl', 'rb') as f:
    data = pickle.load(f)
parameters = data['parameters']
hex_mesh = data['hex_mesh']
hex_cell = hex_mesh.cell
hex_node = hex_mesh.node
mesh_gearx = HexahedronMesh(hex_node, hex_cell)

space_gearx = LagrangeFESpace(mesh_gearx, p=1, ctype='C')

cell_gearx = mesh_gearx.entity('cell')
cell2dof_gearx = space_gearx.cell_to_dof()

u = parameters[..., 0]
v = parameters[..., 1]
w = parameters[..., 2]

u = bm.clip(u, 0, 1)
v = bm.clip(v, 0, 1)
w = bm.clip(w, 0, 1)

bcs_gearxs = [
    (
        # bm.tensor([[1 - u, u]]),
        # bm.tensor([[1 - v, v]]),
        # bm.tensor([[1 - w, w]])
        bm.tensor([[u, 1 - u]]),
        bm.tensor([[v, 1 - v]]),
        bm.tensor([[w, 1 - w]])
    )
    for u, v, w in zip(u, v, w)
]
for idx, (u_tensor, v_tensor, w_tensor) in enumerate(bcs_gearxs):
    u_values = u_tensor.flatten()
    v_values = v_tensor.flatten()
    w_values = w_tensor.flatten()
    print(f"载荷点 {idx + 1} 的重心坐标:\n u,1-u = {u_values}, v,1-v = {v_values}, w,1-w = {w_values}")


phi_gearxs = []
for bcs_gearx in bcs_gearxs:
    phi_gearx = space_gearx.basis(bcs_gearx)
    phi_gearxs.append(phi_gearx)
for idx, phi_gearx in enumerate(phi_gearxs):
    print(f"载荷点 {idx + 1} 处的基函数值:\n", phi_gearx.flatten())
print("------------------------")