
# import mindspore as ms
# import mindspore.numpy as mnp
import torch

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import TriangleMesh

bm.set_backend('mindspore')
# bm.set_backend('pytorch')

NX, NY = 64, 64

mesh = TriangleMesh.from_box(nx=NX, ny=NY)
