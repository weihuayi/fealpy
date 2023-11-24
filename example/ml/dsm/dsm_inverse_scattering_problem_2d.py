import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from scipy.special import hankel1
from scipy.sparse.linalg import spsolve

from fealpy.mesh import UniformMesh2d, TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator, ScalarMassIntegrator, ScalarSourceIntegrator, ScalarConvectionIntegrator, DirichletBC
from fealpy.fem import BilinearForm, LinearForm
from fealpy.pde.diffusion_convection_reaction import PMLPDEModel2d
from fealpy.ml.sampler import CircleCollocator


#定义波数、散射体形状、求解域、网格、入射波
k = 6.5
u_inc = 'cos(d_0*k*x + d_1*k*y) + sin(d_0*k*x + d_1*k*y) * 1j'

def levelset(p):
    ctr = np.array([[0, -1.0]])
    return np.linalg.norm(p - ctr, axis=-1) - 0.3

domain = [-6, 6, -6, 6]
mesh = TriangleMesh.interfacemesh_generator(box=domain, nx=150, ny=150, phi=levelset)
p=1
qf = mesh.integrator(p+3, 'cell')
_, ws = qf.get_quadrature_points_and_weights()
qs = len(ws)

#实例化带PML的pde
pml = PMLPDEModel2d(level_set=levelset,
                 domain=domain,
                 qs=qs,
                 u_inc=u_inc,
                 A=1,
                 k=k, 
                 d=[0, -1], 
                 refractive_index=[1, 1+1/k**2],
                 absortion_constant=1.79,
                 lx=1.0,
                 ly=1.0)

#求解获得散射场数据
space = LagrangeFESpace(mesh, p=p)
space.ftype = complex

D = ScalarDiffusionIntegrator(c=pml.diffusion_coefficient, q=p+3)
C = ScalarConvectionIntegrator(c=pml.convection_coefficient, q=p+3)
M = ScalarMassIntegrator(c=pml.reaction_coefficient, q=p+3)
f = ScalarSourceIntegrator(pml.source, q=p+3)

b = BilinearForm(space)
b.add_domain_integrator([D, C, M])

l = LinearForm(space)
l.add_domain_integrator(f)

A = b.assembly()
F = l.assembly()
bc = DirichletBC(space, pml.dirichlet) 
uh = space.function(dtype=np.complex128)
A, F = bc.apply(A, F, uh)
uh[:] = spsolve(A, F)

#获取指定接收域的散射数据
reciever_points = CircleCollocator(0, 0, 5).run(50)
reciever_points = reciever_points.detach().numpy()
num = mesh.point_to_bc(reciever_points)
location = mesh.location(reciever_points)
u_s_data = np.zeros(len(num), dtype=np.complex128)
for i in range (len(num)):
    u_s_data[i] = uh(num[i])[location[i]]

u_s_data = torch.from_numpy(u_s_data).reshape(-1)

#在指定区域采样
EXTC = 401
HC = 1/EXTC * 4
mesh_col = UniformMesh2d((0, EXTC, 0, EXTC), (HC, HC), origin=(-2, -2))
_bd_node = mesh_col.ds.boundary_node_flag()
sampling_points = torch.from_numpy(mesh_col.entity('node', index=~_bd_node))
reciever_points = torch.from_numpy(reciever_points)

#格林函数
def green_fuc(p1:Tensor, p2:Tensor):

    a = p1[:, None, :] - p2[None, :, :]
    r = torch.sqrt(a[..., 0:1]**2 + a[..., 1:2]**2).view(p1.shape[0], -1)
    val = 1j * hankel1(0, k * r)/4
    return val

#DSM计算数据特征
def phi(p):
    
    gre_func_real = torch.real(green_fuc(reciever_points, p))
    gre_func_imag = torch.imag(green_fuc(reciever_points, p))
    u_s_real = torch.real(u_s_data)
    u_s_imag = torch.imag(u_s_data)
    
    val_1 = torch.sqrt(torch.abs(torch.einsum('i,ij->j', u_s_real, gre_func_real))**2 \
        + torch.abs(torch.einsum('i,ij->j', u_s_imag, gre_func_imag))**2)
    val_2 = torch.sqrt(torch.norm(u_s_real, p=2, dim = 0)**2 + torch.norm(u_s_imag, p=2, dim = 0)**2)
    val_3 = torch.sqrt(torch.norm(gre_func_real, p=2, dim = 0)**2 + torch.norm(gre_func_imag, p=2, dim = 0)**2)
    val_1 = val_1/torch.max(val_1)
    val_2 = val_2/torch.max(val_2)
    val_3 = val_3/torch.max(val_3)
    return val_1/(val_2 * val_3)

#可视化散射场数据以及散射体重建效果
fig_1 = plt.figure()
bc = np.array([[1/3, 1/3, 1/3]], dtype=np.float64)
value = uh(bc)
mesh.add_plot(plt, cellcolor=value[0, ...].real, linewidths=0)
mesh.add_plot(plt, cellcolor=value[0, ...].imag, linewidths=0)
ax_1 = fig_1.add_subplot(1, 3, 1)

mesh.add_plot(ax_1)

ax_1 = fig_1.add_subplot(1, 3, 2, projection='3d')
mesh.show_function(ax_1, np.real(uh))

ax_1 = fig_1.add_subplot(1, 3, 3, projection='3d')
mesh.show_function(ax_1, np.imag(uh))

fig_2, ax_2 = plt.subplots()
scatter = plt.scatter(sampling_points[..., 0], sampling_points[..., 1], c=phi(sampling_points), cmap='inferno', label='Points')
highlighted_circle = plt.Circle((0.0, -1.0), 0.3, color='white', fill=False, linestyle='dashed', linewidth=1)
plt.colorbar(scatter, label='PHI')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Function and Points with Colormap')
ax_2.add_patch(highlighted_circle)

plt.show()
