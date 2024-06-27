import jax
import jax.numpy as jnp
import numpy as np
from fealpy.jax.sph import NodeMesh
from fealpy.jax.sph import SPHSolver
from fealpy.jax.sph.solver import Tag
from fealpy.jax.sph import partition 
from jax_md.partition import Sparse
from fealpy.jax.sph.kernel_function import QuinticKernel
from jax_md import space
from omegaconf import OmegaConf

jax.config.update("jax_enable_x64", True) # 启用 float64 支持
jax.config.update('jax_platform_name', 'cpu')

EPS = jnp.finfo(float).eps
dx = 0.02
h = dx
n_walls = 3 #墙粒子层数
sp = OmegaConf.create({"L": 1.0, "H": 0.2})
dx2n = dx * n_walls * 2 #墙粒子总长度
box_size = np.array([sp.L, sp.H + dx2n])
rho0 = 1.0 #参考密度
V = 1.0 #估算最大流速
v0 = 10.0 #参考速度
c0 = v0 * V #声速
gamma = 1.0
p0 = (rho0 * c0**2)/ gamma #参考压力
p_bg = 5.0 #背景压力,用于计算压力p

#时间步长和步数
g_ext = 2.3 #场外力
eta0 = 0.01 #参考动态粘度
T = 1.5
t_num = int(T / dt)

mesh = NodeMesh.from_heat_transfer_domain()

solver = SPHSolver(mesh)
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.periodic(side=box_size)

#为流体粒子添加噪声
key_prng = jax.random.PRNGKey(123)
key, subkey = jax.random.split(key_prng)
r0_noise = 0.05
noise_std = r0_noise * dx
noise = solver.get_noise_masked(mesh.nodedata["position"].shape, mesh.nodedata["tag"] == Tag.fluid, subkey, std=noise_std)
mesh.nodedata["position"] = shift(mesh.nodedata["position"], noise)
mesh.nodedata["p"] = solver.tait_eos(mesh.nodedata["rho"],c0,rho0,X=p_bg)
print(mesh.nodedata)
