
import torch
from torch import Tensor, tensordot
import numpy as np
from numpy.typing import NDArray

from fealpy.cem.generator import EITDataGenerator, LaplaceDataGenerator2d
from fealpy.mesh import TriangleMesh as TMD
from fealpy.torch.mesh import TriangleMesh
from fealpy.fem import ScalarDiffusionIntegrator as DI_np
from fealpy.torch.fem import ScalarDiffusionIntegrator as DI_torch
from fealpy.torch.fem import BilinearForm


FREQ = [1, 2, 3, 4, 5, 6, 8, 16]
EXT = 8

def level_set(p: NDArray):
    x = p[..., 0]
    y = p[..., 1]
    return x**2 + (y-0.7)**2 - 0.2**2

def neumann(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, "device": points.device}
    theta = torch.arctan2(y, x)
    freq = torch.tensor(FREQ, **kwargs)
    return torch.cos(tensordot(freq, theta, dims=0))

sigma_vals = (10., 1.)

def _coef_func(p: Tensor):
    inclusion = level_set(p) < 0.
    sigma = torch.empty(p.shape[:2], dtype=p.dtype, device=p.device) # (Q, C)
    sigma[inclusion] = sigma_vals[0]
    sigma[~inclusion] = sigma_vals[1]
    return sigma

def _coef_func_np(p: NDArray):
    inclusion = level_set(p) < 0.
    sigma = np.empty(p.shape[:2], dtype=p.dtype) # (Q, C)
    sigma[inclusion] = sigma_vals[0]
    sigma[~inclusion] = sigma_vals[1]
    return sigma


mesh_np = TMD.interfacemesh_generator([-1, 1, -1, 1], nx=EXT, ny=EXT, phi=level_set)
mesh_torch = TriangleMesh.from_numpy(mesh_np)

gen1 = EITDataGenerator(mesh_torch, p=1, q=3)
gen1.set_levelset((10., 1.), level_set)
gen1.set_boundary(neumann, len(FREQ))

gen2 = LaplaceDataGenerator2d.from_cos(
    [-1, 1, -1, 1], EXT, EXT, sigma_vals=(10., 1.),
    levelset=level_set,
    freq=FREQ, phrase=[0.,]
)


# gd1 = gen1.run()[0, :].numpy()
# gd2 = gen2.gd().__next__()


def linf(x, y):
    print(np.max(x), np.max(y))
    return np.max(np.abs(x - y))

# print(linf(gd1, gd2))


# print(c2ip1 == c2ip2)


a1 = DI_torch(_coef_func).assembly(gen1.space)
a2 = DI_np(c=_coef_func_np, q=3).assembly_cell_matrix(gen2.space)
# diff = np.mean(np.abs(a1.numpy() - a2), axis=(1, 2))
# print(np.max(diff))
gdof = gen1.space.number_of_global_dofs()


c2ip1 = gen1.space.cell_to_dof().numpy()
c2ip2 = gen2.space.cell_to_dof()

from scipy.sparse import coo_matrix

bform = BilinearForm(gen1.space)
bform.add_integrator(DI_torch(_coef_func))
A1 = bform.assembly()

def assemble(group_tensor: NDArray, e2dof: NDArray, global_mat_shape):
    I = np.broadcast_to(e2dof[:, :, None], shape=group_tensor.shape)
    J = np.broadcast_to(e2dof[:, None, :], shape=group_tensor.shape)
    return coo_matrix((group_tensor.ravel(), (I.ravel(), J.ravel())), shape=global_mat_shape)

A2 = assemble(a2, c2ip2, (gdof, gdof))


from matplotlib import pyplot as plt

# fig = plt.figure()
# axes = fig.add_subplot(111)
# mesh_np.add_plot(axes, cellcolor=diff, colorbar=True)


fig = plt.figure()
axes = fig.add_subplot(131)
qm = axes.pcolormesh(A1.to_dense().numpy() - gen1._A.to_dense().numpy())
fig.colorbar(qm)

axes = fig.add_subplot(132)
qm = axes.pcolormesh(A2.toarray() - gen2.solver.A_.toarray())
fig.colorbar(qm)

axes = fig.add_subplot(133)
qm = axes.pcolormesh(gen1._A.to_dense().numpy() - gen2.solver.A_.toarray())
fig.colorbar(qm)

plt.show()
