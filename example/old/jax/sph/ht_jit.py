import jax
import jax.numpy as jnp
import numpy as np
from fealpy.jax.mesh.node_mesh import NodeMesh
from fealpy.jax.sph.solver import SPHSolver, Tag
from fealpy.jax.sph import partition 
from fealpy.jax.sph.jax_md.partition import Sparse
from fealpy.jax.sph.kernel_function import QuinticKernel
from jax_md import space
from jax import vmap, jit
import matplotlib.pyplot as plt
import warnings
import time

jax.config.update("jax_enable_x64", True) # 启用 float64 支持
jax.config.update('jax_platform_name', 'cpu')

EPS = jnp.finfo(float).eps
dx = 0.02
h = dx
n_walls = 3 #墙粒子层数
L, H = 1.0, 0.2
dx2n = dx * n_walls * 2 #墙粒子总高度
box_size = np.array([L, H + dx2n])
rho0 = 1.0 #参考密度
V = 1.0 #估算最大流速
v0 = 10.0 #参考速度
c0 = v0 * V #声速
gamma = 1.0
p0 = (rho0 * c0**2)/ gamma #参考压力
p_bg = 5.0 #背景压力,用于计算压力p
tvf = 0 #控制是否使用运输速度
path = "./"

#时间步长和步数
T = 1.5
dt = 0.00045454545454545455
t_num = int(T / dt)

#初始化
mesh = NodeMesh.from_heat_transfer_domain(dx=dx, dy=dx)

solver = SPHSolver(mesh)
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.periodic(side=box_size)

mesh.nodedata["p"] = solver.tait_eos(mesh.nodedata["rho"], c0, rho0, X=p_bg)
mesh.nodedata = solver.boundary_conditions(mesh.nodedata, box_size, dx=dx)

#邻近搜索
neighbor_fn = partition.neighbor_list(
    displacement,
    box_size,
    r_cutoff=kernel.cutoff_radius,
    backend="jaxmd_vmap",
    capacity_multiplier=1.25,
    mask_self=False,
    format=Sparse,
    num_particles_max=mesh.nodedata["position"].shape[0],
    num_partitions=1,
    pbc=np.array([True, True, True]),
)

num_particles = (mesh.nodedata["tag"] != -1).sum()
neighbors = neighbor_fn.allocate(mesh.nodedata["position"], num_particles=num_particles)

start = time.time()
for i in range(100):
    print("i:", i)
    mesh.nodedata["mv"] += 1.0 * dt * mesh.nodedata["dmvdt"]
    mesh.nodedata["tv"] = mesh.nodedata["mv"] + tvf * 0.5 * dt * mesh.nodedata["dtvdt"]
    mesh.nodedata["position"] = shift(mesh.nodedata["position"], 1.0 * dt * mesh.nodedata["tv"])

    num_particles = (mesh.nodedata["tag"] != -1).sum()
    neighbors = neighbors.update(mesh.nodedata["position"], num_particles=num_particles)

    i_s, j_s = neighbors.idx
    i_s0 = [(len(mesh.nodedata["position"]) - 1) if x == len(mesh.nodedata["position"]) else x for x in i_s]
    j_s0 = [(len(mesh.nodedata["position"]) - 1) if x == len(mesh.nodedata["position"]) else x for x in j_s]
    r_i_s, r_j_s = mesh.nodedata["position"][jnp.array(i_s0)], mesh.nodedata["position"][jnp.array(j_s0)]
    dr_i_j = vmap(displacement)(r_i_s, r_j_s)
    dist = space.distance(dr_i_j)
    w_dist = vmap(kernel)(dist)

    e_s = dr_i_j / (dist[:, None] + EPS)
    grad_w_dist_norm = vmap(kernel.grad_value)(dist)
    grad_w_dist = grad_w_dist_norm[:, None] * e_s

    #外加速度场
    g_ext = solver.external_acceleration(mesh.nodedata["position"], box_size, dx=dx)

    #标记
    wall_mask = jnp.where(jnp.isin(mesh.nodedata["tag"], jnp.array([1, 3])), 1.0, 0.0)
    fluid_mask = jnp.where(mesh.nodedata["tag"] == 0, 1.0, 0.0)

    #密度处理
    rho_summation = solver.compute_rho(mesh.nodedata["mass"], i_s, w_dist)
    rho = jnp.where(fluid_mask, rho_summation, mesh.nodedata["rho"])

    #计算压力和背景压力
    p = solver.tait_eos(rho,c0,rho0,X=p_bg)
    pb = solver.tait_eos(jnp.zeros_like(p),c0,rho0,X=p_bg)

    #边界处理
    p, rho, mv, tv, T = solver.enforce_wall_boundary(mesh.nodedata, p, g_ext, i_s, j_s, w_dist, dr_i_j, with_temperature=True)
    mesh.nodedata["rho"] = rho
    mesh.nodedata["mv"] = mv
    mesh.nodedata["tv"] = tv

    #计算下一步的温度导数
    T += dt * mesh.nodedata["dTdt"]
    mesh.nodedata["T"] = T
    mesh.nodedata["dTdt"] = solver.temperature_derivative(mesh.nodedata, kernel, e_s, dr_i_j, dist, i_s, j_s, grad_w_dist_norm)

    #更新动量速度的加速度
    mesh.nodedata["dmvdt"] = solver.compute_mv_acceleration(mesh.nodedata, i_s, j_s, dr_i_j, dist, grad_w_dist_norm, p)
    mesh.nodedata["dmvdt"] = mesh.nodedata["dmvdt"] + g_ext
    mesh.nodedata["p"] = p

    #更新运输速度的加速度
    mesh.nodedata["dtvdt"] = solver.compute_tv_acceleration(mesh.nodedata, i_s, j_s, dr_i_j, dist, grad_w_dist_norm, pb)

    #更新边界条件
    mesh.nodedata = solver.boundary_conditions(mesh.nodedata, box_size, dx=dx)

end = time.time()
print(end - start)