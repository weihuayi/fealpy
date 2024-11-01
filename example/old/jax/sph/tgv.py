import jax
jax.config.update("jax_enable_x64", True) # 启用 float64 支持
#jax.config.update('jax_platform_name', 'cpu')
import enum
import jax.numpy as jnp
import numpy as np
from fealpy.jax.mesh.node_mesh import NodeMesh
from fealpy.jax.sph import SPHSolver,TimeLine
from fealpy.jax.sph import partition 
from fealpy.jax.sph.jax_md.partition import Sparse
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

#初始化
mesh = NodeMesh.from_tgv_domain(box_size, dx)
solver = SPHSolver(mesh)
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.periodic(side=box_size) #TODO
timeline = TimeLine

#邻近搜索
neighbor_fn = partition.neighbor_list(
    displacement,
    box_size,
    r_cutoff=QuinticKernel(h=h, dim=2).cutoff_radius,
    backend="jaxmd_vmap",
    capacity_multiplier=1.25,
    mask_self=False,
    format=Sparse,
    num_particles_max=mesh.nodedata["position"].shape[0],
    num_partitions=mesh.nodedata["position"].shape[0],
    pbc=[True, True, True],
)
##数据类型？？
warnings.filterwarnings("ignore", category=FutureWarning)
neighbors = neighbor_fn.allocate(mesh.nodedata["position"], num_particles=mesh.nodedata["position"].shape[0])

for i in range(t_num+2):
    
    mesh.nodedata['mv'] += 1.0*dt*mesh.nodedata["dmvdt"]
    mesh.nodedata['tv'] = mesh.nodedata['mv']
    #mesh.nodedata['tv'] = mesh.nodedata['mv'] + 0.5*dt*mesh.nodedata['dtvdt']
    mesh.nodedata["position"] = shift(mesh.nodedata["position"], 1.0 * dt * mesh.nodedata["tv"])

    neighbors = neighbors.update(mesh.nodedata["position"], num_particles=mesh.nodedata["position"].shape[0])
    
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
