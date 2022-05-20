import numpy as np
import matplotlib.pyplot as plt
import taichi as ti

from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.ti import TriangleMesh # 基于 Taichi 的三角形网格

ti.init()

p = 2
node, cell = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri', returnnc=True)
mesh = TriangleMesh(node, cell)
print(mesh.node)
