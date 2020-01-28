import numpy as np
from fealpy.mesh import  TriangleMesh
import matplotlib.pyplot as plt


class TriangleMeshTest:

    def init_mesh(self,h=1):
        node = np.array([
            (0, 0),
            (h, 0),
            (h, h),
            (0, h)], dtype=np.float)
        cell = np.array([
            (1, 2, 0),
            (3, 0, 2)], dtype=np.int)
        tmesh = TriangleMesh(node, cell)
        return tmesh

    def test_to_quadmesh(self):
        mesh = self.init_mesh(h=1)
        qmesh = mesh.to_quadmesh()
        return qmesh


test = TriangleMeshTest()
mesh = test.test_to_quadmesh()
mesh.uniform_refine(n=1)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()

