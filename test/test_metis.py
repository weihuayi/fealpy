import sys
import numpy as np
from meshpy.triangle import MeshInfo, build
from fealpy.mesh.TriangleMesh import TriangleMesh 
from fealpy.graph import metis
import matplotlib.pyplot as plt

import scipy.io as sio

h = float(sys.argv[1])
n = int(sys.argv[2])

mesh_info = MeshInfo()

# Set the vertices of the domain [0, 1]^2
mesh_info.set_points([
    (0,0), (1,0), (1,1), (0,1)])

# Set the facets of the domain [0, 1]^2
mesh_info.set_facets([
    [0,1],
    [1,2],
    [2,3],
    [3,0]
    ])

# Generate the tet mesh
mesh = build(mesh_info, max_volume=(h)**2)

point = np.array(mesh.points, dtype=np.float)
cell = np.array(mesh.elements, dtype=np.int)

tmesh = TriangleMesh(point, cell)

# Partition the mesh cells into n parts 
edgecuts, parts = metis.part_mesh(tmesh, nparts=n, entity='cell', contig=True)

node = tmesh.node
edge = tmesh.ds.edge
cell = tmesh.ds.cell
cell2edge = tmesh.ds.cell_to_edge()
edge2cell = tmesh.ds.edge_to_cell()
isBdNode = tmesh.ds.boundary_node_flag()

data = {'Point':node, 'Face':edge+1, 'Elem':cell+1,
        'Edge2Elem':edge2cell+1, 'isBdPoint':isBdNode, 'Partitions':parts+1}

sio.matlab.savemat('test'+str(n)+'parts'+str(h)+'.mat', data)

fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes, cellcolor='w')
tmesh.find_node(axes, color=parts, markersize=20)
fig.savefig('test.pdf')
plt.show()
