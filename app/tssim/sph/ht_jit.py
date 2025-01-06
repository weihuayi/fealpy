import jax
import jax.numpy as jnp
import numpy as np
from fealpy.mesh.node_mesh import NodeMesh
from fealpy.cfd.sph.particle_solver import SPHSolver, Tag, TimeLoop
from fealpy.cfd.sph import partition 
from fealpy.cfd.sph.jax_md.partition import Sparse
from fealpy.cfd.sph.particle_kernel_function import QuinticKernel
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

forward = solver.forward_wrapper_ht(displacement, kernel, box_size, dx, c0, rho0, p_bg, dt)

advance = TimeLoop(forward, shift, tvf=tvf)
advance = jit(advance)

start = time.time()
for i in range(1000):
    print(i)
    mesh.nodedata, neighbors = advance(dt, mesh.nodedata, neighbors) 
    #fname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'
    #solver.write_vtk(mesh.nodedata, fname)

end = time.time()
print(end-start)