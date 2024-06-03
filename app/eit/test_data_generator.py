"""An example generating a single data for EIT.

This can only test if the generator can solve the PDE, but can not test if
the gd and gn saved are correct.
"""

import torch
from torch import Tensor, tensordot, rand
from matplotlib import pyplot as plt

from fealpy.torch.mesh import TriangleMesh
from fealpy.mesh import TriangleMesh as TMD
from data_generator import EITDataGenerator

seed = 2024
kwargs = {"dtype": torch.float64, "device": 'cpu'}
NUM_CIR = 3
SIGMA = [10., 1.]
FREQ = [1, 2, 3, 4, 5, 6, 8, 16]
EXT = 63


def levelset(p: Tensor, centers: Tensor, radius: Tensor):
    """
    Calculate level set function value.
    """
    struct = p.shape[:-1]
    p = p.reshape(-1, p.shape[-1])
    dis = torch.norm(p[:, None, :] - centers[None, :, :], dim=-1) # (N, NCir)
    ret, _ = torch.min(dis - radius[None, :], dim=-1) # (N, )
    return ret.reshape(struct)

def neumann(points: Tensor):
    x = points[..., 0]
    y = points[..., 1]
    kwargs = {'dtype': points.dtype, "device": points.device}
    theta = torch.arctan2(y, x)
    freq = torch.tensor(FREQ, **kwargs)
    return torch.cos(tensordot(freq, theta, dims=0))

torch.manual_seed(seed)
mesh = TriangleMesh.from_box((-1, 1, -1, 1), EXT, EXT, ftype=torch.float64, device='cpu')
generator = EITDataGenerator(mesh=mesh)
gn = generator.set_boundary(neumann, batched=True)

ctrs = rand(NUM_CIR, 2, **kwargs) * 1.6 - 0.8 # (NCir, GD)
b, _ = torch.min(0.9-torch.abs(ctrs), axis=-1) # (NCir, )
rads = rand(NUM_CIR, **kwargs) * (b-0.1) + 0.1 # (NCir, )
ls_fn = lambda p: levelset(p, ctrs, rads)

label = generator.set_levelset(SIGMA, ls_fn)
uh = generator.run(return_full=True)


### visualize

fig = plt.figure(figsize=(12, 6))
fig.tight_layout()
fig.suptitle('Parallel solving Poisson equation on 2D Triangle mesh')
mesh_numpy = TMD.from_box((-1, 1, -1, 1), EXT, EXT)
value = generator.space.value(uh, torch.tensor([[1/3, 1/3, 1/3]], device='cpu', dtype=torch.float64)).squeeze(0)
value = value.cpu().numpy()

for i in range(8):
    axes = fig.add_subplot(2, 4, i+1)
    mesh_numpy.add_plot(axes, cellcolor=value[i, :], cmap='jet', linewidths=0, showaxis=True)

plt.show()
