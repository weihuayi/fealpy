
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
mesh.uniform_refine(n=1)

node = mesh.entity('node')
cell = mesh.entity('cell')

arrayprint("node", node)
arrayprint("cell", cell)

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True, fontsize=32)
mesh.find_cell(axes, showindex=True, fontsize=38)
plt.show()
