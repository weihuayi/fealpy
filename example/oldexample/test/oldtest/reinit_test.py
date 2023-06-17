import numpy as np
import sys


from fealpy.mesh.level_set_function import dcircle
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh 
from fealpy.mesh.interface_mesh_generator import find_cut_point 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pandas as pd

def pandas_groupby_minimum(idx, val):
    df = pd.DataFrame({'ID' : idx, 'val' : val})
    return df.groupby('ID')['val'].min().values

def circle(p, cxy, r):
    x = p[:, 0]
    y = p[:, 1]
    return  ((x - 1)**2 + (y - 1)**2 + 1 )*(np.sqrt((x - cxy[0])**2 + (y - cxy[1])**2) - r)

def sign(phi):
    eps = 1e-8
    sign = np.sign(phi)
    sign[np.abs(phi) < eps] = 0
    return sign

def HG(mesh, phi, phiSign, h):

    N = mesh.number_of_points()
    NE = mesh.number_of_edges()
    nx = mesh.ds.nx 
    ny = mesh.ds.ny
    dx = mesh.dx
    dy = mesh.dy


    dx0 = np.zeros(N, dtype=np.float)
    dx1 = np.zeros(N, dtype=np.float)
    dy0 = np.zeros(N, dtype=np.float)
    dy1 = np.zeros(N, dtype=np.float)


    edge = mesh.ds.edge

    isXEdge = np.zeros(NE, dtype=np.bool_)
    isXEdge[ny*(nx+1):] = True
    
    dx0[edge[isXEdge, 0]] = (phi[edge[isXEdge, 1]] - phi[edge[isXEdge, 0]])/dx
    dx1[edge[isXEdge, 1]] = dx0[edge[isXEdge, 0]] 

    dy0[edge[~isXEdge, 0]] = (phi[edge[~isXEdge, 1]] - phi[edge[~isXEdge, 0]])/dy
    dy1[edge[~isXEdge, 1]] = dy0[edge[~isXEdge, 0]]

    dx0[-ny-1:] = dx0[-2*(ny+1):-ny-1]
    dx1[0:ny+1] = dx1[ny+1:2*(ny+1)]

    dy0[ny::ny+1] = dy0[ny-1:-1:ny+1]
    dy1[0:-ny:ny+1] = dy1[1:-ny+1:ny+1]

    dxx = (dx0 - dx1)/dx
    dyy = (dy0 - dy1)/dx

    isCutEdge = phiSign[edge[:, 0]]*phiSign[edge[:, 1]] < 0
    dx0[edge[isXEdge & isCutEdge, 0]] = -phi[edge[isXEdge & isCutEdge, 0]]/h[isXEdge[isCutEdge]]
    dx1[edge[isXEdge & isCutEdge, 1]] = phi[edge[isXEdge & isCutEdge, 1]]/(dx - h[isXEdge[isCutEdge]])
    dy0[edge[~isXEdge & isCutEdge, 0]] = -phi[edge[~isXEdge & isCutEdge, 0]]/h[~isXEdge[isCutEdge]]
    dy1[edge[~isXEdge & isCutEdge, 1]] = phi[edge[~isXEdge & isCutEdge, 1]]/(dy - h[~isXEdge[isCutEdge]])

    minmodx0 = np.zeros(N, dtype=np.float)
    minmodx1 = np.zeros(N, dtype=np.float)
    minmody0 = np.zeros(N, dtype=np.float)
    minmody1 = np.zeros(N, dtype=np.float)
    minmodx0[edge[isXEdge, 0]] = np.minimum(np.abs(dxx[edge[isXEdge, 0]]), np.abs(dxx[edge[isXEdge, 1]]))
    minmodx1[edge[isXEdge, 1]] = minmodx0[edge[isXEdge, 0]]
    minmody0[edge[~isXEdge, 0]] = np.minimum(np.abs(dyy[edge[~isXEdge, 0]]), np.abs(dyy[edge[~isXEdge, 1]]))
    minmody1[edge[~isXEdge, 1]] = minmody0[edge[~isXEdge, 0]] 

    a = dx0 - 0.5*dx*minmodx0
    b = dx1 + 0.5*dx*minmodx1
    c = dy0 - 0.5*dy*minmody0
    d = dy1 + 0.5*dy*minmody1

    idx = edge[isXEdge & isCutEdge, 0]
    a[idx] = dx0[idx] - 0.5*h[isXEdge[isCutEdge]]*minmodx0[idx]
    idx = edge[isXEdge & isCutEdge, 1]
    b[idx] = dx1[idx] + 0.5*(dx - h[isXEdge[isCutEdge]])*minmodx1[idx]

    idx = edge[~isXEdge & isCutEdge, 0]
    c[idx] = dy0[idx] - 0.5*h[~isXEdge[isCutEdge]]*minmody0[idx]
    idx = edge[~isXEdge & isCutEdge, 1]
    d[idx] = dy1[idx] + 0.5*(dy - h[~isXEdge[isCutEdge]])*minmody1[idx]

    hg = np.zeros(N, dtype=np.float)

    hg[phiSign>=0] = np.sqrt(np.maximum(np.minimum(a[phiSign>=0], 0)**2, np.maximum(b[phiSign>=0], 0)**2) 
            + np.maximum(np.minimum(c[phiSign>=0], 0)**2, np.maximum(d[phiSign>=0], 0)**2))
    hg[phiSign < 0] = np.sqrt(np.maximum(np.maximum(a[phiSign<0], 0)**2, np.minimum(b[phiSign<0], 0)**2) 
            + np.maximum(np.maximum(c[phiSign<0], 0)**2, np.minimum(d[phiSign<0], 0)**2))

    return hg


n = int(sys.argv[1])
nx = n
ny = n

box = [-2, 2, -2, 2]
cxy = (0.0, 0.0)
r = 1 
interface0 = lambda p: dcircle(p, cxy, r)

interface = lambda p: circle(p, cxy, r)


mesh = StructureQuadMesh(box, nx, ny)

dx = mesh.dx
dy = mesh.dy

point = mesh.point
N = point.shape[0]
phi = interface(point)

phiSign = sign(phi)

edge = mesh.ds.edge
NE = edge.shape[0]
isCutEdge = phiSign[edge[:, 0]]*phiSign[edge[:, 1]] < 0
A = point[edge[isCutEdge, 0]].copy()
B = point[edge[isCutEdge, 1]].copy()
cutPoint = find_cut_point(interface, A, B)
isCutPoint = np.zeros(N, dtype=np.bool_)
isCutPoint[edge[isCutEdge]] = True

isXEdge = np.zeros(NE, dtype=np.bool_)
isXEdge[ny*(nx+1):] = True

A = point[edge[isCutEdge, 0]]
h = np.zeros(A.shape[0], dtype=np.float)
h[isXEdge[isCutEdge]] = cutPoint[isXEdge[isCutEdge], 0] - A[isXEdge[isCutEdge], 0]
h[~isXEdge[isCutEdge]] = cutPoint[~isXEdge[isCutEdge], 1] - A[~isXEdge[isCutEdge], 1]

val = np.zeros((NE, 2), dtype=np.float)
val[isXEdge, :] = dx
val[~isXEdge, :] = dy

val[isXEdge & isCutEdge, 0] = h[isXEdge[isCutEdge]]
val[isXEdge & isCutEdge, 1] = dx - h[isXEdge[isCutEdge]]
val[~isXEdge & isCutEdge, 0] = h[~isXEdge[isCutEdge]]
val[~isXEdge & isCutEdge, 1] = dy - h[~isXEdge[isCutEdge]]

dt = 0.45*pandas_groupby_minimum(edge.flatten(), val.flatten())

d = interface0(point) 


for i in range(500):
    phi1 = phi - dt*phiSign*(HG(mesh, phi, phiSign, h) - 1)
    phi2 = phi1 - dt*phiSign*(HG(mesh, phi1, phiSign, h) - 1)
    phi = (phi1 + phi2)/2
    print("Near interface:", np.max(np.abs(phi - d)[np.abs(phi) < 1.2*dx]))
    print("Whole error:", np.max(np.abs(phi - d)[phi > -0.8]))

fig = plt.figure()
ax = fig.gca(projection='3d')
X = point[:, 0].reshape(nx+1, ny+1)
Y = point[:, 1].reshape(nx+1, ny+1)
Z = phi.reshape(nx+1, ny+1) - d.reshape(nx+1, ny+1)
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig = plt.figure()
axes = fig.gca() 
mesh.add_plot(axes, cellcolor=[0.5, 0.9, 0.45])
mesh.find_point(axes, point=cutPoint, markersize=30)
plt.show()
