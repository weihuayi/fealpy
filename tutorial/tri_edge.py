
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh

def arrayprint(name, a):
    """
    Note
    ----
    打印一个名字为 name 的数组 a，每一行之前加一个行号
    """
    print("\n", name+":")
    for (i, row) in enumerate(a):
        print(i, ": ", row)

node = np.array([
    [0.0, 0.0], # 0 
    [1.0, 0.0], # 1 
    [1.0, 1.0], # 2
    [0.0, 1.0]  # 3
    ], dtype=np.float64)

cell = np.array([
    [1, 2, 0], # 0
    [3, 0, 2], # 1
    ], dtype=np.int_)


mesh = TriangleMesh(node, cell)
#mesh.uniform_refine(n=1)

node = mesh.entity('node')
cell = mesh.entity('cell')

# 从 cell 出发，构造 edge、edge2cel、cell2edge
NC = mesh.number_of_cells() #
NEC = 3 # 每个单元有 3 条边 
localEdge = np.array([
    [1, 2], # 局部 0 号边
    [2, 0], # 局部 1 号边
    [0, 1], # 局部 2 号边
    ], dtype=np.int_) # (3, 2)
# （NC, 3)---> (NC, 3, 2) --> (3*NC, 2)
totalEdge = cell[:, localEdge].reshape(-1, 2) 
stotalEdge = np.sort(totalEdge, axis=-1)

sedge, i0, j = np.unique(stotalEdge, 
        axis=0,
        return_index=True, 
        return_inverse=True) 

i1 = np.zeros_like(i0)
i1[j] = range(NEC*NC)

edge = totalEdge[i0] # 
cell2edge = j.reshape(-1, 3) # (NC, 3)

NE = len(edge)
edge2cell = np.zeros((NE, 4), dtype=np.int_)
edge2cell[:, 0] = i0//NEC
edge2cell[:, 1] = i1//NEC
edge2cell[:, 2] = i0%NEC
edge2cell[:, 3] = i1%NEC



arrayprint("totalEdge:", totalEdge)

# sedge[j] == stotalEdge
arrayprint("stotalEdge:", stotalEdge)
arrayprint("sedge[j]", sedge[j])
arrayprint("j", j)
arrayprint("cell2edge", cell2edge)

arrayprint("sedge:", sedge)
# sedge == stotalEdge[i0]
arrayprint("stotalEdge[i0]", stotalEdge[i0])
arrayprint("i0", i0)
arrayprint("i1", i1)

arrayprint("node", node) # (NN, 2)
arrayprint("cell", cell) # (NC, 3)
arrayprint("edge", edge) # (NC, 3)
arrayprint("edge2cell", edge2cell) # (NE, 4)
#edge = mesh.entity('edge')
#arrayprint("edge", edge) # (NE, 2)
#cell2edge = mesh.ds.cell_to_edge()
#arrayprint("cell2edge", cell2edge) # (NC, 3)
#edge2cell = mesh.ds.edge_to_cell()
#arrayprint("edge2cell", edge2cell) # (NE, 4)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=28)
mesh.find_edge(axes, showindex=True, fontsize=30)
mesh.find_cell(axes, showindex=True, fontsize=32)
plt.show()
