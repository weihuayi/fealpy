import jax
jax.config.update("jax_enable_x64", True) # 启用 float64 支持
#jax.config.update('jax_platform_name', 'cpu')
import enum
import jax.numpy as jnp
import numpy as np
from fealpy.jax.sph import NodeMesh
from fealpy.jax.sph import SPHSolver
from fealpy.jax.sph import partition 
from jax_md.partition import Sparse
from fealpy.jax.sph.kernel_function import QuinticKernel
from jax_md import space
from jax import ops, vmap
from jax import lax
import matplotlib.pyplot as plt
import warnings


EPS = jnp.finfo(float).eps
dx = 0.02
dy = 0.02
h = dx #平滑长度 实际上是3dx
Vmax = 1.0 #预期最大速度
c0 =10 * Vmax #声速
rho0 = 1.0 #参考密度
eta0 = 0.01
nu = 1.0 #运动粘度
T = 2 #终止时间
dt = 0.0004 #时间间隔
t_num = int(T / dt)
dim = 2 #维数
box_size = jnp.array([1.0,1.0]) #模拟区域
path = "./"


mesh = NodeMesh.from_tgv_domain(box_size, dx)
solver = SPHSolver(mesh)
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.periodic(side=box_size) #TODO

#nodedata初始化
position = mesh.node
NN = position.shape[0]
tag = jnp.full(len(position), 0, dtype=int)
mv = jnp.zeros((NN, 2), dtype=jnp.float64)
tv = jnp.zeros((NN, 2), dtype=jnp.float64)
x = position[:,0]
y = position[:,1]
u0 = -jnp.cos(2.0 * jnp.pi * x) * jnp.sin(2.0 * jnp.pi * y)
v0 = jnp.sin(2.0 * jnp.pi * x) * jnp.cos(2.0 * jnp.pi *y)
mv = mv.at[:,0].set(u0)
mv = mv.at[:,1].set(v0)
tv = mv
volume = jnp.ones(NN, dtype=jnp.float64) * dx * dy
rho = jnp.ones(NN, dtype=jnp.float64) * rho0
mass = jnp.ones(NN, dtype=jnp.float64) * dx * dy * rho0
eta = jnp.ones(NN, dtype=jnp.float64) * eta0

mesh.nodedata = {
    "position": position,
    "tag": tag,
    "mv": mv,
    "tv": tv,
    "dmvdt": jnp.zeros_like(mv),
    "dtvdt": jnp.zeros_like(mv),
    "rho": rho,
    "mass": mass,
    "eta": eta,
}

#邻近搜索
neighbor_fn = partition.neighbor_list(
    displacement,
    box_size,
    r_cutoff=QuinticKernel(h=h, dim=2).cutoff_radius,
    backend="jaxmd_vmap",
    capacity_multiplier=1.25,
    mask_self=False,
    format=Sparse,
    num_particles_max=NN,
    num_partitions=NN,
    pbc=[True, True, True],
)
##数据类型？？
warnings.filterwarnings("ignore", category=FutureWarning)
neighbors = neighbor_fn.allocate(mesh.nodedata["position"], num_particles=NN)

for i in range(t_num+2):
    
    mesh.nodedata['mv'] += 1.0*dt*mesh.nodedata["dmvdt"]
    mesh.nodedata['tv'] = mesh.nodedata['mv']
    #mesh.nodedata['tv'] = mesh.nodedata['mv'] + 0.5*dt*mesh.nodedata['dtvdt']
    mesh.nodedata["position"] = shift(mesh.nodedata["position"], 1.0 * dt * mesh.nodedata["tv"])

    neighbors = neighbors.update(mesh.nodedata["position"], num_particles=NN)
    
    r = mesh.nodedata["position"]
    i_s, j_s = neighbors.idx
    r_i_s, r_j_s = r[i_s], r[j_s]
    dr_i_j = vmap(displacement)(r_i_s, r_j_s)
    dist = space.distance(dr_i_j)
    w_dist = vmap(kernel.value)(dist)  
    e_s = dr_i_j / (dist[:, None] + EPS) 
    grad_w_dist_norm = vmap(kernel.grad_value)(dist)
    grad_w_dist = grad_w_dist_norm[:, None] * e_s


    mesh.nodedata['rho'] = solver.compute_rho(mesh.nodedata['mass'], i_s, w_dist)
    p = solver.tait_eos(mesh.nodedata['rho'],c0,rho0)
    background_pressure_tvf = solver.tait_eos(jnp.zeros_like(p), c0, rho0)
    
    mesh.nodedata["dmvdt"] = solver.compute_mv_acceleration(\
            mesh.nodedata, i_s, j_s, dr_i_j, dist, grad_w_dist_norm, p)
    #dtvdt = solver.compute_tv_acceleration(state, i_s,j_s, grad_w_dist)
    
    #fname = path + 'test_'+ str(i+1).zfill(10) + '.h5'
    #solver.write_h5(mesh.nodedata, fname)
    #fname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'
    #solver.write_vtk(mesh.nodedata, fname)
