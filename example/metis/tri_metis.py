#!/usr/bin/env python3
# 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh 
from fealpy.graph import metis

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        利用 Metis 分割三角形网格 
        """)

parser.add_argument('--h',
        default=0.1, type=float,
        help='生成网格单元尺寸, 默认 0.1.')

parser.add_argument('--n',
        default=4, type=int,
        help='网格分割块数.')


args = parser.parse_args()

h = args.h
n = args.n

vertices = np.array([ (0,0), (1,0), (1,1), (0,1)], dtype=np.float64)
mesh = TriangleMesh.from_polygon_gmsh(vertices, h)

# Partition the mesh cells into n parts 
edgecuts, parts = metis.part_mesh(mesh, nparts=n, entity='node')

node = mesh.node
edge = mesh.ds.edge
cell = mesh.ds.cell
cell2edge = mesh.ds.cell_to_edge()
edge2cell = mesh.ds.edge_to_cell()
isBdNode = mesh.ds.boundary_node_flag()

data = {'Point':node, 'Face':edge+1, 'Elem':cell+1,
        'Edge2Elem':edge2cell+1, 'isBdPoint':isBdNode, 'Partitions':parts+1}

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor='w')
mesh.find_node(axes, color=parts, markersize=20)
fig.savefig('test.pdf')
plt.show()
