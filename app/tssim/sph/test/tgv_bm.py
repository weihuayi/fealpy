from fealpy.backend import backend_manager as bm
from fealpy.mesh.node_mesh import NodeMesh
from fealpy.cfd.sph.particle_solver_new import SPHSolver, Space
from fealpy.cfd.sph.particle_kernel_function_new import QuinticKernel

bm.set_backend("pytorch")

import torch   #打印
import jax.numpy as jnp
torch.set_printoptions(precision=8, sci_mode=False)
torch.set_printoptions(threshold=jnp.inf)
from jax import vmap   #Vmap

EPS = bm.finfo(float).eps
dx = 0.02
dy = dx
h = dx 
Vmax = 1.0 #预期最大速度
c0 =10 * Vmax #声速
rho0 = 1.0 #参考密度
eta0 = 0.01
nu = 1.0 #运动粘度
T = 2 #终止时间
dt = 0.0004 #时间间隔
t_num = int(T / dt)
dim = 2 #维数
box_size = bm.array([1.0,1.0], dtype=bm.float64) #模拟区域
path = "./"

mesh = NodeMesh.from_tgv_domain(box_size, dx)
solver = SPHSolver(mesh)
space = Space()
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.periodic(side=box_size)

node_self, neighbors = bm.query_point(mesh.nodedata["position"], mesh.nodedata["position"], 3*h, box_size, True, [True, True, True])

for i in range(200):
    print(i)
    mesh.nodedata['mv'] += 1.0*dt*mesh.nodedata["dmvdt"]
    mesh.nodedata['tv'] = mesh.nodedata['mv']
    mesh.nodedata["position"] = shift(mesh.nodedata["position"], 1.0 * dt * mesh.nodedata["tv"])

    r = mesh.nodedata["position"]
    node_self, neighbors = bm.query_point(r, r, 3*h, box_size, True, [True, True, True])

    r_i_s, r_j_s = r[neighbors], r[node_self]
    dr_i_j = bm.vmap(displacement)(r_i_s, r_j_s)
    dist = space.distance(dr_i_j)
    w_dist = bm.vmap(kernel.value)(dist)

    e_s = dr_i_j / (dist[:, None] + EPS) # (dr/dx,dr/dy)
    grad_w_dist_norm = bm.vmap(kernel.grad_value)(dist)
    grad_w_dist = grad_w_dist_norm[:, None] * e_s

    mesh.nodedata['rho'] = solver.compute_rho(mesh.nodedata['mass'], neighbors, w_dist)
    p = solver.tait_eos(mesh.nodedata['rho'], c0, rho0)
    background_pressure_tvf = solver.tait_eos(bm.zeros_like(p), c0, rho0)

    mesh.nodedata["dmvdt"] = solver.compute_mv_acceleration(\
            mesh.nodedata, neighbors, node_self, dr_i_j, dist, grad_w_dist_norm, p)

    #fname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'
    #solver.write_vtk(mesh.nodedata, fname)
    