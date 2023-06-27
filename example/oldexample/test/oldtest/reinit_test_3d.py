import numpy as np
import sys


from fealpy.mesh.level_set_function import Sphere
from fealpy.mesh.StructureHexMesh import StructureHexMesh
from fealpy.mesh.interface_mesh_generator import find_cut_point 

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import pandas as pd

def pandas_groupby_minimum(idx, val):
    df = pd.DataFrame({'ID' : idx, 'val' : val})
    return df.groupby('ID')['val'].min().values

def sphere(p, cxy, r):
    x = p[:, 0]
    y = p[:, 1]
    z = p[:, 2]
    return  (x - cxy[0])**2 + (y - cxy[1])**2 + (z - cxy[2])**2 - r*r

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
    nz = mesh.ds.nz
    dx = (box[1] - box[0])/nx
    dy = (box[3] - box[2])/ny
    dz = (box[5] - box[4])/nz


    dx0 = np.zeros(N, dtype=np.float)
    dx1 = np.zeros(N, dtype=np.float)
    dy0 = np.zeros(N, dtype=np.float)
    dy1 = np.zeros(N, dtype=np.float)
    dz0 = np.zeros(N, dtype=np.float)
    dz1 = np.zeros(N, dtype=np.float)

    edge = mesh.ds.edge
    
    isZEdge = np.zeros(NE, dtype=np.bool_)
    isYEdge = np.zeros(NE, dtype=np.bool_)
    isXEdge = np.zeros(NE, dtype=np.bool_)

    N0 = 0
    N1 = nz*(ny+1)*(nx+1)
    isZEdge[N0:N1] = True
    N0 = N1
    N1 = N0 + ny*(nx+1)*(nz+1)
    isYEdge[N0:N1] = True
    isXEdge[N1:] = True

    
    dx0[edge[isXEdge, 0]] = (phi[edge[isXEdge, 1]] - phi[edge[isXEdge, 0]])/dx
    dx1[edge[isXEdge, 1]] = dx0[edge[isXEdge, 0]]

    dy0[edge[isYEdge, 0]] = (phi[edge[isYEdge, 1]] - phi[edge[isYEdge, 0]])/dy
    dy1[edge[isYEdge, 1]] = dy0[edge[isYEdge, 0]] 
    
    dz0[edge[isZEdge, 0]] = (phi[edge[isZEdge, 1]] - phi[edge[isZEdge, 0]])/dz
    dz1[edge[isZEdge, 1]] = dz0[edge[isZEdge, 0]] 

    dx0[-(nz+1)*(ny+1):] = dx0[-2*(nz+1)*(ny+1):-(nz+1)*(ny+1)]
    dx1[0:(nz+1)*(ny+1)] = dx1[(nz+1)*(ny+1):2*(nz+1)*(ny+1)]

    # 原来这里的程序有错误
    xm = dx0.reshape(nx+1, ny+1, nz+1)
    xm[-1, :, :] = xm[-2, :, :]
    xm = dx1.reshape(nx+1, ny+1, nz+1)
    xm[0, :, :] = xm[1, :, :]

    ym = dy0.reshape(nx+1, ny+1, nz+1)
    ym[:, -1, :] = ym[:, -2, :]
    ym = dy1.reshape(nx+1, ny+1, nz+1)
    ym[:, 0, :] = ym[:, 1, :]

    zm = dz0.reshape(nx+1, ny+1, nz+1)
    zm[:, :, -1] = zm[:, :, -2]
    zm = dz1.reshape(nx+1, ny+1, nz+1)
    zm[:, :, 0] = zm[:, :, 1]

    dxx = (dx0 - dx1)/dx
    dyy = (dy0 - dy1)/dy
    dzz = (dz0 - dz1)/dz

    isCutEdge = phiSign[edge[:, 0]]*phiSign[edge[:, 1]] < 0
    dx0[edge[isXEdge & isCutEdge, 0]] = -phi[edge[isXEdge & isCutEdge,0]]/h[isXEdge[isCutEdge]]
    dx1[edge[isXEdge & isCutEdge, 1]] = phi[edge[isXEdge & isCutEdge, 1]]/(dx - h[isXEdge[isCutEdge]])
    dy0[edge[isYEdge & isCutEdge, 0]] = -phi[edge[isYEdge & isCutEdge,0]]/h[isYEdge[isCutEdge]]
    dy1[edge[isYEdge & isCutEdge, 1]] = phi[edge[isYEdge & isCutEdge, 1]]/(dy - h[isYEdge[isCutEdge]])
    dz0[edge[isZEdge & isCutEdge, 0]] = -phi[edge[isZEdge & isCutEdge,0]]/h[isZEdge[isCutEdge]]
    dz1[edge[isZEdge & isCutEdge, 1]] = phi[edge[isZEdge & isCutEdge, 1]]/(dz - h[isZEdge[isCutEdge]])
    
    minmodx0 = np.zeros(N, dtype=np.float)
    minmodx1 = np.zeros(N, dtype=np.float)
    minmody0 = np.zeros(N, dtype=np.float)
    minmody1 = np.zeros(N, dtype=np.float)
    minmodz0 = np.zeros(N, dtype=np.float)
    minmodz1 = np.zeros(N, dtype=np.float)
   
    minmodx0[edge[isXEdge, 0]] = np.minimum(np.abs(dxx[edge[isXEdge, 0]]), np.abs(dxx[edge[isXEdge, 1]]))
    minmodx1[edge[isXEdge, 1]] = minmodx0[edge[isXEdge, 0]] 
    minmody0[edge[isYEdge, 0]] = np.minimum(np.abs(dyy[edge[isYEdge, 0]]), np.abs(dyy[edge[isYEdge, 1]]))
    minmody1[edge[isYEdge, 1]] = minmody0[edge[isYEdge, 0]] 
    minmodz0[edge[isZEdge, 0]] = np.minimum(np.abs(dzz[edge[isZEdge, 0]]), np.abs(dzz[edge[isZEdge, 1]]))
    minmodz1[edge[isZEdge, 1]] = minmodz0[edge[isZEdge, 0]]

    a = dx0 - 0.5*dx*minmodx0
    b = dx1 + 0.5*dx*minmodx1
    c = dy0 - 0.5*dy*minmody0
    d = dy1 + 0.5*dy*minmody1
    e = dz0 - 0.5*dz*minmodz0
    f = dz1 + 0.5*dz*minmodz1
    
    idx = edge[(isXEdge & isCutEdge), 0]
    a[idx] = dx0[idx] - 0.5*h[isXEdge[isCutEdge]]*minmodx0[idx]
    idx = edge[isXEdge & isCutEdge, 1]
    b[idx] = dx1[idx] + 0.5*(dx - h[isXEdge[isCutEdge]])*minmodx1[idx]
    
    idx = edge[isYEdge & isCutEdge, 0]
    c[idx] = dy0[idx] - 0.5*h[isYEdge[isCutEdge]]*minmody0[idx]
    idx = edge[isYEdge & isCutEdge, 1]
    d[idx] = dy1[idx] + 0.5*(dy - h[isYEdge[isCutEdge]])*minmody1[idx]
    
    idx = edge[isZEdge & isCutEdge, 0]
    e[idx] = dz0[idx] - 0.5*h[isZEdge[isCutEdge]]*minmodz0[idx]
    idx = edge[isZEdge & isCutEdge, 1]
    f[idx] = dz1[idx] + 0.5*(dz - h[isZEdge[isCutEdge]])*minmodz1[idx]

    hg = np.zeros(N, dtype=np.float)

    hg[phiSign>=0] = np.sqrt(np.maximum(np.minimum(a[phiSign>=0], 0)**2, np.maximum(b[phiSign>=0], 0)**2) 
            + np.maximum(np.minimum(c[phiSign>=0], 0)**2, np.maximum(d[phiSign>=0], 0)**2)
            + np.maximum(np.minimum(e[phiSign>=0], 0)**2, np.maximum(f[phiSign>=0], 0)**2))
    hg[phiSign < 0] = np.sqrt(np.maximum(np.maximum(a[phiSign<0], 0)**2, np.minimum(b[phiSign<0], 0)**2) 
            + np.maximum(np.maximum(c[phiSign<0], 0)**2, np.minimum(d[phiSign<0], 0)**2)
            + np.maximum(np.maximum(e[phiSign<0], 0)**2, np.minimum(f[phiSign<0], 0)**2))

    return hg


n = int(sys.argv[1])
nx = n
ny = n
nz = n

box = [-2, 2, -2, 2, -2, 2]
cxy = (0.0, 0.0, 0.0)
r = 1 
interface0 = Sphere(cxy, r)

interface = lambda p: sphere(p, cxy, r)


mesh = StructureHexMesh(box, nx, ny, nz)

dx = (box[1] - box[0])/nx
dy = (box[3] - box[2])/ny
dz = (box[5] - box[4])/nz

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
isXEdge[-nx*(ny+1)*(nz+1):] = True
isYEdge = np.zeros(NE, dtype=np.bool_)
isYEdge[ny*(nz+1)*(nx+1):ny*(nz+1)*(nx+1)+nx*(ny+1)*(nz+1)] = True
isZEdge = np.zeros(NE, dtype=np.bool_)
isZEdge[0:nz*(nx+1)*(ny+1)] = True

A = point[edge[isCutEdge, 0]]
h = np.zeros(A.shape[0], dtype=np.float)
h[isXEdge[isCutEdge]] = cutPoint[isXEdge[isCutEdge], 0] - A[isXEdge[isCutEdge], 0]
h[isYEdge[isCutEdge]] = cutPoint[isYEdge[isCutEdge], 1] - A[isYEdge[isCutEdge], 1]
h[isZEdge[isCutEdge]] = cutPoint[isZEdge[isCutEdge], 2] - A[isZEdge[isCutEdge], 2]
print(h)

val = np.zeros((NE, 2), dtype=np.float)
val[isXEdge, :] = dx
val[isYEdge, :] = dy
val[isZEdge, :] = dz

val[isXEdge & isCutEdge, 0] = h[isXEdge[isCutEdge]]
val[isXEdge & isCutEdge, 1] = dx - h[isXEdge[isCutEdge]]
val[isYEdge & isCutEdge, 0] = h[isYEdge[isCutEdge]]
val[isYEdge & isCutEdge, 1] = dy - h[isYEdge[isCutEdge]]
val[isZEdge & isCutEdge, 0] = h[isZEdge[isCutEdge]]
val[isZEdge & isCutEdge, 1] = dz - h[isZEdge[isCutEdge]]
dt = 0.3*pandas_groupby_minimum(edge.flatten(), val.flatten())

d = interface0(point) 
for i in range(100):
    phi1 = phi - dt*phiSign*(HG(mesh, phi, phiSign, h) - 1)
    phi2 = phi1 - dt*phiSign*(HG(mesh, phi1, phiSign, h) - 1)
    phi = (phi1 + phi2)/2
    print("Near interface:", np.max(np.abs(phi - d)[np.abs(phi) < 1.2*dx]))
    print("Whole error:", np.max(np.abs(phi - d)[phi > -0.8]))

