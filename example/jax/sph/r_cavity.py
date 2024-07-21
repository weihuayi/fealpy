import jax
import jax.numpy as jnp
import numpy as np
from fealpy.jax.mesh.node_mesh import NodeMesh
from fealpy.jax.sph import SPHSolver
from fealpy.jax.sph.solver import Tag
from fealpy.jax.sph import partition 
from fealpy.jax.sph.jax_md.partition import Sparse
from fealpy.jax.sph.kernel_function import QuinticKernel
from jax_md import space
from jax import vmap
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True) # 启用 float64 支持
jax.config.update('jax_platform_name', 'cpu')

EPS = jnp.finfo(float).eps
dx = 0.5
dy = dx
h = 1.5 * dx
rho0 = 737.54 
n = 0.3083
tau_s = 16834.4
mu0 = 938.4118
c1 = 0.0894
B = 5.914e7
eta = 0.5 #动量方程中的参数
dt = 1e-7

#计算区域
box_size = jnp.array([[0, 50], [-3*dx, 5 + 4*dx]])

#初始化
mesh = NodeMesh.from_long_rectangular_cavity_domain(dx=dx)
solver = SPHSolver(mesh)
kernel = QuinticKernel(h=h, dim=2)

#新增粒子物理量
new_node = jnp.mgrid[0:dx:dx, dy:5:dy].reshape(2, -1).T
new_tag = jnp.full((new_node.shape[0],), 0, dtype=int)
new_mv = jnp.tile(jnp.array([5, 0]), (new_node.shape[0], 1))
new_rho = jnp.ones(new_node.shape[0]) * rho0
new_mass = jnp.ones(new_node.shape[0]) * dx * dy * rho0
mesh.add_node_data(['position', 'tag', 'mv', 'tv', 'dmvdt', 'dtvdt', 'drhodt', 'rho', 'p', 'mass'],
                    [new_node, new_tag, new_mv,new_mv,jnp.zeros_like(new_mv),jnp.zeros_like(new_mv),
                    jnp.zeros_like(new_rho), new_rho, jnp.zeros_like(new_rho), new_mass])
