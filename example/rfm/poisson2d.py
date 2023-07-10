
import torch
from fealpy.pinn.modules import RandomFeature
from fealpy.mesh import UniformMesh2d

mesh = UniformMesh2d((0, 1, 0, 1), h=(0.1, 0.1), origin=(0, 0))
nodes = torch.from_numpy(mesh.node)
edge_length = 0.1

model = RandomFeature(50, nodes, edge_length)
