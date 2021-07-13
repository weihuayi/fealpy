#!/usr/bin/env python3
# 
import sys

import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import HalfEdgeMesh2d, CVTPMesher

from scipy.spatial import voronoi_plot_2d


# 正方形
node = np.array([
    ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0)],dtype=np.float)
edge = np.array([
    (0, 1),(1, 2),(2, 3),(3, 0)], dtype=np.int)
subdomain = np.array([
    (1, 0),(1, 0),(1, 0),(1, 0)], dtype=np.int)
mesh = HalfEdgeMesh2d.from_edges(node, edge, subdomain)
uniform_mesh = CVTPMesher(mesh)
uniform_mesh.uniform_meshing(n=2)
vor, start = uniform_mesh.voronoi()

i =0
while i<10:
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
voronoi_plot_2d(vor, ax=axes,show_vertices = False)
plt.show()


# 正方形, 中间有圆形的洞
n = 20
h = 2*np.pi/n
theta = np.arange(0, 2*np.pi, h)
vertices1 = np.array([(-2.0,-2.0),(2.0,-2.0),(2.0,2.0),(-2.0,2.0)])
vertices2 = np.zeros((n, 2), dtype=np.float)
vertices2[:, 0] = np.cos(theta)
vertices2[:, 1] = np.sin(theta)
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
subdomain[4:,0] = -1
subdomain[4:1] = 1
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



