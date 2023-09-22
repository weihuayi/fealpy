"""
RFM for the Hemker model
Equation:
 -\\varepsilon\\Delta u + \\bm{b}\cdot\\nabla u = f
"""

import torch
from torch import Tensor
from torch.nn import init
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from fealpy.mesh import UniformMesh2d
from fealpy.ml.modules import (
    RandomFeatureSpace, PoUSpace, Cos, PoUSin, Function,
    Solution
)
from fealpy.ml.sampler import Collocator, CircleCollocator

PI = torch.pi

ep = torch.tensor(0.01, dtype=torch.float64)
b = torch.tensor([1.0, 0.0], dtype=torch.float64)

def boundary_left(p: Tensor):
    return torch.zeros((p.shape[0], 1), dtype=p.dtype)

def boundary_t_b_r(p: Tensor):
    return torch.zeros((p.shape[0], 1), dtype=p.dtype)

def boundary_circle(p: Tensor):
    return torch.ones((p.shape[0], 1), dtype=p.dtype)

def source(p: Tensor):
    return torch.zeros((p.shape[0], 1), dtype=p.dtype)

Jn = 50

mesh = UniformMesh2d((0, 6, 0, 3), (2, 2), origin=(-3, -3))
node = torch.from_numpy(mesh.entity('node'))

def factory(i: int):
    if i in {5, 6, 9, 10}:
        sp = RandomFeatureSpace(2, Jn*4, Cos(), bound=(16.0, PI))
        init.normal_(sp.frequency, 0, 8)
    elif i in {0, 1, 2, 3}:
        sp = RandomFeatureSpace(2, 8, Cos(), bound=(0.5, PI))
        init.normal_(sp.frequency, 0, 0.5)
    else:
        sp = RandomFeatureSpace(2, Jn, Cos(), bound=(4.0, PI))
        init.normal_(sp.frequency, 0, 4.0)
    return sp

space = PoUSpace(factory, centers=node, radius=1.0, pou=PoUSin(), print_status=True)


### generate collocation points

N = 200
col_in = Collocator([-3, 9, -3, 3], [2*N, N]).run()
col_in = col_in[col_in[:, 0]**2 + col_in[:, 1]**2 > 1, :]

col_left = Collocator([-3, -3, -3, 3], [1, N]).run()
col_top = Collocator([-3, 9, 3, 3], [N*2, 1]).run()
col_btm = Collocator([-3, 9, -3, -3], [N*2, 1]).run()
col_right = Collocator([9, 9, -3, 3], [1, N]).run()
col_cir = CircleCollocator().run(N)

b_ = torch.cat([source(col_in),
                boundary_left(col_left),
                boundary_t_b_r(col_top),
                boundary_t_b_r(col_btm),
                boundary_t_b_r(col_right),
                boundary_circle(col_cir)], dim=0)

diffusion = -ep * space.laplace_basis(col_in)
convection = space.convect_basis(col_in, coef=b)
del col_in
value_left = space.basis(col_left)
del col_left
grad_value_top = space.convect_basis(col_top, coef=torch.tensor([0, 1], dtype=torch.float64))
del col_top
grad_value_btm = space.convect_basis(col_btm, coef=torch.tensor([0, -1], dtype=torch.float64))
del col_btm
grad_value_right = space.convect_basis(col_right, coef=torch.tensor([1, 0], dtype=torch.float64))
del col_right
value_cir = space.basis(col_cir)
del col_cir

A_ = torch.cat([(diffusion + convection)/64,
                value_left,
                grad_value_top/8,
                grad_value_btm/8,
                grad_value_right/8,
                value_cir], dim=0)

del diffusion, convection, value_left, grad_value_top, grad_value_btm, grad_value_right, value_cir

A_ = csr_matrix(A_.detach().cpu().numpy())
b_ = csr_matrix(b_.detach().cpu().numpy())

um = spsolve(A_.T@A_, A_.T@b_)
del A_, b_

model = Function(space, 1, torch.from_numpy(um))


from matplotlib import pyplot as plt
fig = plt.figure("RFM for Hemker")
axes = fig.add_subplot(111)

class RemoveCircle(Solution):
    def forward(self, p: Tensor) -> Tensor:
        flag = p[:, 0]**2 + p[:, 1]**2 < 1
        ret = super().forward(p)
        ret[flag] = 0.0
        return ret

qm = RemoveCircle(model).add_pcolor(axes, [-3, 9, -3, 3], [400, 200])
axes.set_aspect('equal')
fig.colorbar(qm)

fig = plt.figure("LOADS")
axes = fig.add_subplot(221)
axes.scatter(space.partitions[1].space.frequency[:, 0], um[space.partition_basis_slice(1)])
axes.set_title("PART-1")
axes = fig.add_subplot(222)
axes.scatter(space.partitions[5].space.frequency[:, 0], um[space.partition_basis_slice(5)])
axes.set_title("PART-5")
axes = fig.add_subplot(223)
axes.scatter(space.partitions[9].space.frequency[:, 0], um[space.partition_basis_slice(9)])
axes.set_title("PART-9")
axes = fig.add_subplot(224)
axes.scatter(space.partitions[13].space.frequency[:, 0], um[space.partition_basis_slice(13)])
axes.set_title("PART-13")
plt.show()
