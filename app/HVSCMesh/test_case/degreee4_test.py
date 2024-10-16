import numpy as np
import matplotlib.pyplot as plt
from fealpy.old.mesh import TriangleMesh

node = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0,1.0],
                 [0.5, 0.5], [2.0, 0.0], [2.0, 1.0], [1.5,0.5]], dtype=np.float64)
cell = np.array([[0,1,4], [1,2,4], [2,3,4], [3,0,4], [1,5,7],
                 [5,6,7], [6,2,7], [2,1,7]],dtype=np.int64)

mesh = TriangleMesh(node, cell)
mesh.delete_degree_4()

print(mesh.entity('node'))
print(mesh.entity('cell'))

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()
