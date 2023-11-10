"""
RFM for the Hemker model
Equation:
 -\\varepsilon\\Delta u + \\bm{b}\cdot\\nabla u = f
"""

import torch
from torch import Tensor
from torch.nn import init

from fealpy.mesh import UniformMesh2d
from fealpy.ml.modules import (
    RandomFeatureSpace, PoUSpace, Cos, PoUSin, Solution
)
from fealpy.ml.sampler import Collocator, CircleCollocator
from fealpy.ml.operators import (
    ScalerDiffusion, ScalerConvection, ScalerMass, Form
)

PI = torch.pi
ep = torch.tensor(0.1, dtype=torch.float64)
b = torch.tensor([1.0, 0.0], dtype=torch.float64)


def boundary_circle(p: Tensor):
    return torch.ones((p.shape[0], 1), dtype=p.dtype)


Jn = 64

def factory(i: int):
    if i in {5, 6, 9, 10}:
        sp = RandomFeatureSpace(2, Jn*4, Cos(), bound=(8.0, PI))
        init.normal_(sp.frequency, 0, 8.0)
    elif i in {0, 1, 2, 3}:
        sp = RandomFeatureSpace(2, 8, Cos(), bound=(0.5, PI))
        init.normal_(sp.frequency, 0, 0.5)
    else:
        sp = RandomFeatureSpace(2, Jn, Cos(), bound=(4.0, PI))
        init.normal_(sp.frequency, 0, 4.0)
    return sp

mesh = UniformMesh2d((0, 6, 0, 3), (2, 2), origin=(-3, -3))
space = PoUSpace.from_uniform_mesh(factory, mesh, 'node', pou=PoUSin(), print_status=True)

### generate collocation points

N = 200
col_in = Collocator([-3, 9, -3, 3], [2*N, N]).run()
col_in = col_in[col_in[:, 0]**2 + col_in[:, 1]**2 > 1, :]

col_left = Collocator([-3, -3, -3, 3], [1, N]).run()
col_top = Collocator([-3, 9, 3, 3], [N*2, 1]).run()
col_btm = Collocator([-3, 9, -3, -3], [N*2, 1]).run()
col_right = Collocator([9, 9, -3, 3], [1, N]).run()
col_cir = CircleCollocator().run(N)

form = Form(space)
form.add(col_in, (ScalerDiffusion(ep), ScalerConvection(b)))
form.add(col_left, ScalerMass())
form.add(col_right, ScalerConvection([1., 0.]))
form.add(col_btm, ScalerConvection([0., -1.]))
form.add(col_top, ScalerConvection([0., 1.]))
form.add(col_cir, ScalerMass(), 1.0)

model = form.spsolve(rescale=100, ridge=1e-6)
um = model.um

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
