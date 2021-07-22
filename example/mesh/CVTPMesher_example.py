#!/usr/bin/env python3
# 

import argparse
import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeMesh2d, CVTPMesher

from scipy.spatial import voronoi_plot_2d


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        复杂二维区域上的 CVT  多边形网格生成示例。
        """)

parser.add_argument('--domain',
        default='square', type=str,
        help='区域类型, 默认是 square, 还可以选择：.')

parser.add_argument('--nlloyd',
        default=100, type=int,
        help='lloyd 算法迭代次数, 默认迭代 100 次.')

parser.add_argument('--nbrefine',
        default=2, type=int,
        help='区域边界的加密次数，默认迭代 2 次.')

args = parser.parse_args()

domain = args.domain
nlloyd = args.nlloyd
nbrefine = args.nbrefine


if domain == 'square':
    node = np.array([
        ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0)],dtype=np.float)
    edge = np.array([
        (0, 1),(1, 2),(2, 3),(3, 0)], dtype=np.int)
    subdomain = np.array([
        (1, 0),(1, 0),(1, 0),(1, 0)], dtype=np.int)
    mesh = HalfEdgeMesh2d.from_edges(node, edge, subdomain)
    times = None
else:
    raise ValueError("the domain argument appear error!") 

mesher = CVTPMesher(mesh)
mesher.uniform_meshing(nb=nbrefine) 
vor, start = mesher.voronoi()

i = 0
while i < nlloyd:
    vor = mesher.lloyd_opt(vor, start)
    i+=1

plt.show()
fig = plt.figure()
axes = fig.gca()
mesh.add_halfedge_plot(axes, showindex=False)
mesh.find_node(axes, showindex=False)
plt.show()

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, color='k', showindex=False)
mesh.find_node(axes, node=vor.points, showindex=False)
voronoi_plot_2d(vor, ax=axes,show_vertices = False)
plt.show()


# 正方形, 中间有圆形的洞
n = 20
h = 2*np.pi/n
theta = np.arange(0, 2*np.pi, h)
vertices1 = np.array([(-2.0,-2.0),(2.0,-2.0),(2.0,2.0),(-2.0,2.0)])
vertices2 = np.zeros((n, 2), dtype=np.float)
vertices2[:, 0] = np.cos(theta)
vertices2[:, 1] = -np.sin(theta)
vertices = np.vstack((vertices1,vertices2))
n = len(vertices)
fixed = np.ones(n, dtype=np.bool_)
fixed[4:] = False
facets = np.zeros((n,2), dtype=np.int)
facets[:4, 0] = range(0, 4)
facets[:3, 1] = range(1, 4)
facets[4:, 0] = range(4, n)
facets[4:-1, 1] = range(5, n)
facets[-1,1] = 4
subdomain = np.zeros((n, 2),dtype=np.int)
subdomain[:4, 0] = 1
subdomain[4:,0] = 1
subdomain[4:1] = -1
times = np.zeros(n)
times[:4] = 5
times[4:] = 2
mesh = HalfEdgeMesh2d.from_edges(vertices, facets, subdomain,
        fixed)
uniform_mesh = CVTPMesher(mesh)
uniform_mesh.uniform_meshing(n=5,times = times)

vor, start = uniform_mesh.voronoi()

i =0
while i<20:
    vor = uniform_mesh.Lloyd(vor,start)
    i+=1

plt.show()
fig = plt.figure()
axes = fig.gca()
mesh.add_halfedge_plot(axes, showindex=False)
mesh.find_node(axes, showindex=False)
plt.show()

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, color='k', showindex=False)
mesh.find_node(axes, node=vor.points, showindex=False)
voronoi_plot_2d(vor, ax=axes,show_vertices = False,point_size = 0.1)
plt.show()

pmesh = uniform_mesh.to_PolygonMesh(vor)
fig = plt.figure()
axes = fig.gca()
pmesh.add_plot(axes)
plt.show()



