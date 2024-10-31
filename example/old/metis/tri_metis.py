#!/usr/bin/env python3
# 
import argparse # 参数解析
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

parser.add_argument('--etype',
        default='node', type=str,
        help='网格分割类型.')



args = parser.parse_args()

h = args.h
n = args.n
etype = args.etype

vertices = np.array([ (0,0), (1,0), (1,1), (0,1)], dtype=np.float64)
mesh = TriangleMesh.from_polygon_gmsh(vertices, h)

# Partition the mesh cells into n parts 
edgecuts, parts = metis.part_mesh(mesh, nparts=n, entity=etype)

node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor='w')
if etype == 'node':
    mesh.find_node(axes, color=parts, markersize=20)
elif etype == 'cell':
    mesh.find_cell(axes, color=parts, markersize=20)
fig.savefig('test.pdf')
plt.show()
