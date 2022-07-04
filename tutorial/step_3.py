
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory as MF

box = [0, 1, 0, 1]
mesh = MF.boxmesh2d(box, nx=1, ny=1, meshtype='tri')

node = mesh.entity('node')
cell = mesh.entity('cell')

NC = mesh.number_of_cells()

print("node:\n", node)
print("cell:\n", cell)

localEdge = np.array([
    [1, 2], # 0 
    [2, 0], # 1
    [0, 1], # 2
    ], dtype=np.int_) # (3, 2)

totalEdge = cell[:, localEdge].reshape(-1, 2)  # (NC, 3, 2) --> (3*NC, 2)
print(totalEdge)
stotalEdge = np.sort(totalEdge, axis=-1)
print(stotalEdge)

_, i0, j = np.unique(stotalEdge, 
        return_index=True, return_inverse=True, axis=0)

# edge == totalEdge[i0, :]
edge = totalEdge[i0, :]
NE = edge.shape[0]

# edge[j, :] == totalEdge
# len(j) == 3*NC
cell2edge = j.reshape(NC, 3)# (NC, 3)

i1 = np.zeros_like(i0)
i1[j] = range(3*NC)

edge2cell = np.zeros((NE, 4), dtype=np.int_) # (NE, 4)
edge2cell[:, 0] = i0//3  # 左边单元对应的全局编号
edge2cell[:, 1] = i1//3 # 右边单元对应的全局编号
edge2cell[:, 2] = i0%3 # 在左边单元中的局部编号 
edge2cell[:, 3] = i1%3# 在右边单元中的局部编号

print(edge2cell)
print(edge)

cell2node = mesh.ds.cell_to_node()
node2cell = mesh.ds.node_to_cell()
node2edge = mesh.ds.node_to_edge()


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=25)
mesh.find_edge(axes, showindex=True, fontsize=30)
mesh.find_cell(axes, showindex=True, fontsize=50)
plt.show()
