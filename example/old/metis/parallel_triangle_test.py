"""
Parallel Hello World
"""

import sys
import numpy as np
import matplotlib.pyplot as plt 
from mpi4py import MPI
import pyparmetis 


from fealpy.graph import metis
from meshpy.triangle import MeshInfo, build
from fealpy.mesh.TriangleMesh import TriangleMesh


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
name = MPI.Get_processor_name()

mesh_info = MeshInfo()
mesh_info.set_points([(0,0), (1,0), (1,1), (0,1)])
mesh_info.set_facets([[0,1], [1,2], [2,3], [3,0]])  
h = 0.05
mesh = build(mesh_info, max_volume=h**2)
node = np.array(mesh.points, dtype=np.float)
cell = np.array(mesh.elements, dtype=np.int)

NC = cell.shape[0]

n = NC//size
r = NC%size
elmdist = np.zeros(size + 1, dtype=np.int)
elmdist[1:] = n
elmdist[1:r+1] += 1
elmdist = np.add.accumulate(elmdist)

eptr = np.arange(0, 3*(elmdist[rank+1] - elmdist[rank]+1), 3)
eind = cell[elmdist[rank]:elmdist[rank+1]].flatten()

edgecuts, part = pyparmetis.part_mesh(5, elmdist, eptr, eind, comm)


tmesh = TriangleMesh(node, cell)
tmesh.celldata['cell_process_id'] = part
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes, cellcolor='w', markersize=200)
tmesh.find_cell(axes, color=tmesh.celldata['cell_process_id']) 
plt.show()

