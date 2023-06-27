
import numpy as np
import matplotlib.pyplot as plt

from fealpy.mesh.StructureQuadMesh import StructureQuadMesh

L=20
box = [0, L, 0, L]
qmesh = StructureQuadMesh(box, nx=2, ny=2)
C = qmesh.ds.peoriod_matrix()
print(C.toarray())

fig = plt.figure()
axes = fig.gca()
qmesh.add_plot(axes)
qmesh.find_node(axes, showindex=True)
plt.show()
