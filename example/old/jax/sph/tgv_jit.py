import jax
#jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True) # 启用 float64 支持
import enum
import jax.numpy as jnp
import numpy as np
from fealpy.jax.mesh.node_mesh import NodeMesh
from fealpy.jax.sph import SPHSolver,TimeLine
from fealpy.jax.sph import partition 
from fealpy.jax.sph.jax_md.partition import Sparse
from fealpy.jax.sph.kernel_function import QuinticKernel
#from fealpy.jax.sph.jax_md import space
from jax_md import space
from jax import ops, vmap
from jax import lax, jit
import matplotlib.pyplot as plt
import warnings
import time


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
forward = solver.forward_wrapper(displacement, kernel)

advance = TimeLine(forward, shift)
advance = jit(advance)
##数据类型？？
warnings.filterwarnings("ignore", category=FutureWarning)
neighbors = neighbor_fn.allocate(mesh.nodedata["position"], num_particles=mesh.nodedata["position"].shape[0])

start = time.time()
for i in range(100):
    print(i)
    mesh.nodedata, neighbors = advance(dt, mesh.nodedata, neighbors) 
    #fname = path + 'test_'+ str(i+1).zfill(10) + '.vtk'
    #solver.write_vtk(mesh.nodedata, fname)

end = time.time()
print(end-start)
