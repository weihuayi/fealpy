
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh

# 网格节点数组
node = np.array([[0.0, 0.0],
                 [1.0, 0.0],
                 [1.0, 1.0],
                 [0.0, 1.0]], dtype=np.float)
# 网格单元数组
cell = np.array([[1, 2, 0],
                 [3, 0, 2]], dtype=np.int32)

mesh = TriangleMesh(node, cell)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)

nodeIMatrix, cellIMatrix = mesh.uniform_refine(returnim=True)
print(nodeIMatrix[0].toarray())
print(cellIMatrix[0].toarray())
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_cell(axes, showindex=True)

plt.show()

