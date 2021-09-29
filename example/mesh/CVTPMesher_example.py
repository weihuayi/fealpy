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
        help='区域类型, 默认是 square, 还可以选择：Lshape, circle,triangle,circle_h,square_h,partition_tr,partition_s')

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
        ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0)],dtype=np.float64)
    edge = np.array([
        (0, 1),(1, 2),(2, 3),(3, 0)], dtype=np.int_)
    subdomain = np.array([
        (1, 0),(1, 0),(1, 0),(1, 0)], dtype=np.int_)
    mesh = HalfEdgeMesh2d.from_edges(node, edge, subdomain)
    fixed = None
    adaptive = False

elif domain == 'Lshape':
    node = np.array([
        ( 0.0, 0.0),( 1.0, 0.0),( 1.0,  1.0),(0.0,  1.0),
        (-1.0, 1.0),(-1.0, 0.0),(-1.0, -1.0),(0.0, -1.0)],dtype=np.float64)
    fixed = np.array([1, 1, 1, 0, 1, 0, 1, 1],dtype=np.bool_)
    edge = np.array([
       (0, 1), (1, 2), (2, 3), (3, 4), 
       (4, 5), (5, 6), (6, 7), (7, 0)],dtype=np.int_)
    subdomain = np.array([
       (1, 0), (1, 0), (1, 0), (1, 0),
       (1, 0), (1, 0), (1, 0), (1, 0)],dtype=np.int_)
    mesh = HalfEdgeMesh2d.from_edges(node, edge, subdomain, fixed)
    adaptive = False

elif domain =='circle':
    n = 20
    h = 2*np.pi/n
    theta = np.arange(0, 2*np.pi, h)
    node = np.zeros((n, 2), dtype=np.float64)
    node[:, 0] = np.cos(theta)
    node[:, 1] = np.sin(theta)
    fixed = np.zeros(n, dtype=np.bool_)
    edge = np.zeros((n,2), dtype=np.int_)
    edge[:, 0] = range(0, n)
    edge[:-1, 1] = range(1, n)
    subdomain = np.zeros((n, 2),dtype=np.int_)
    subdomain[:, 0] = 1
    mesh = HalfEdgeMesh2d.from_edges(node, edge, subdomain)
    adaptive = False

elif domain =='triangle':
    # n>=3
    node = np.array([
        ( 0.0, 0.0),( 2.0, 0.0),( 1.0, 1.0)],dtype=np.float_)
    edge = np.array([
        (0, 1),(1, 2),(2, 0)], dtype=np.int_)
    subdomain = np.array([
        (1, 0),(1, 0),(1, 0)], dtype=np.int_)
    mesh = HalfEdgeMesh2d.from_edges(node, edge, subdomain)
    adaptive = True
    fixed = None

elif domain =='circle_h':
    n = 20
    h = 2*np.pi/n
    theta = np.arange(0, 2*np.pi, h)
    node1 = np.array([(-2.0,-2.0),(2.0,-2.0),(2.0,2.0),(-2.0,2.0)])
    node2 = np.zeros((n, 2), dtype=np.float64)
    node2[:, 0] = np.cos(theta)
    node2[:, 1] = -np.sin(theta)
    node = np.vstack((node1,node2))
    n = len(node)
    fixed = np.ones(n, dtype=np.bool_)
    fixed[4:] = False
    edge = np.zeros((n,2), dtype=np.int_)
    edge[:4, 0] = range(0, 4)
    edge[:3, 1] = range(1, 4)
    edge[4:, 0] = range(4, n)
    edge[4:-1, 1] = range(5, n)
    edge[-1,1] = 4
    subdomain = np.zeros((n, 2),dtype=np.int_)
    subdomain[:4, 0] = 1
    subdomain[4:,0] = 1
    subdomain[4:,1] = -1
    mesh = HalfEdgeMesh2d.from_edges(node, edge, subdomain,fixed)
    adaptive = True

elif domain == 'square_h':    
    node = np.array([
        ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0, 1.0),
        ( 0.4, 0.4),( 0.4, 0.8),( 0.8, 0.8),( 0.8, 0.4)],dtype=np.float64)
    edge = np.array([
        (0, 1),(1, 2),( 2, 3),( 3, 0),
        (4, 5),(5, 6),( 6, 7),( 7, 4)], dtype=np.int_)
    subdomain = np.array([
        (1, 0),(1, 0),(1, 0),(1, 0),
        (1,-1),(1,-1),(1,-1),(1,-1)], dtype=np.int_)

    mesh = HalfEdgeMesh2d.from_edges(node, edge, subdomain)
    adaptive = True
    fixed = None

elif domain == 'partition_tr':
    vertices = np.array([
        ( 0.0, 0.0),( 1.0, 0.0),( 1.0, 1.0),( 0.0,
            1.0),(0.5,0.5)],dtype=np.float64)
    facets = np.array([
        (0, 1),(1, 2),(2, 3),(3, 0),
        (0, 4),(4, 3),(4, 1),(4, 2)], dtype=np.int_)
    subdomain = np.array([
        (1, 0),(2, 0),(3, 0),(4, 0),
        (4, 1),(4, 3),(2, 1),(3, 2)], dtype=np.int_)
    mesh = HalfEdgeMesh2d.from_edges(vertices, facets, subdomain)
    adaptive = True
    fixed = None

elif domain == 'partition_s':
    vertices = np.array([
        ( 0.0, 0.0),(0.5, 0.0),( 1.0, 0.0),(1.0, 0.5),
        ( 1.0, 1.0),(0.5, 1.0),( 0.0, 1.0),(0.0, 0.5),
        ( 0.5, 0.5)],dtype=np.float64)
    facets = np.array([
        (0, 1),(1, 2),(2, 3),(3, 4),
        (4, 5),(5, 6),(6, 7),(7, 0),
        (1, 8),(8, 7),(8, 3),(8, 5)], dtype=np.int_)
    subdomain = np.array([
        (1, 0),(2, 0),(2, 0),(3, 0),
        (3, 0),(4, 0),(4, 0),(1, 0),
        (1, 2),(1, 4),(3, 2),(4, 3)], dtype=np.int_)
    mesh = HalfEdgeMesh2d.from_edges(vertices, facets, subdomain)
    adaptive = False
    fixed = None

else:
    raise ValueError("the domain argument appear error!") 

mesher = CVTPMesher(mesh,fixed)
vor = mesher.voronoi_meshing(nb=nbrefine,adaptive = adaptive)

i = 0
while i < nlloyd:
    vor = mesher.lloyd_opt(vor)
    i+=1

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, color='k', showindex=False)
mesh.find_node(axes, node=vor.points, showindex=False)
voronoi_plot_2d(vor, ax=axes,show_vertices = False)
plt.show()

pmesh = mesher.to_polygonMesh(vor)
fig = plt.figure()
axes= fig.gca()
pmesh.add_plot(axes)
plt.show()
