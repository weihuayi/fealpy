import numpy as np
import matplotlib.pyplot as plt
import taichi as ti

from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.ti import TriangleMesh 

ti.init()

node, cell = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri', returnnc=True)
mesh = TriangleMesh(node, cell)

window = ti.ui.Window('Triangle Mesh', (640, 360))

while window.running:
    mesh.add_plot(window)
    window.show()
