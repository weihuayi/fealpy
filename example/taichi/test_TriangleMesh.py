import numpy as np
import matplotlib.pyplot as plt
import taichi as ti

from fealpy.mesh import MeshFactory as MF
from TriangleMesh import TriangleMesh # 基于 Taichi 的三角形网格
from wei_lagrange import LagrangeFEMSpace2d
from fealpy.functionspace import LagrangeFiniteElementSpace as LFESpace

ti.init()

p = 2
node, cell = MF.boxmesh2d([0, 1, 0, 1], nx=1, ny=1, meshtype='tri', returnnc=True)
mesh = TriangleMesh(node, cell)
b = mesh.gglambda
d = mesh.glambda
c = b.to_numpy()
space = LagrangeFEMSpace2d(2)
a = space.stiff_matrix(2,2)
A = np.einsum('cmn,ijmn->cij',c,a)

print(A)
