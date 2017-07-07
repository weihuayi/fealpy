import numpy as np

from fealpy.mesh.level_set_function import dcircle
from fealpy.mesh.StructureQuadMesh import StructureQuadMesh 
from fealpy.mesh.interface_mesh_generator import find_cut_point 

def circle(p, cxy, r):
    x = p[:, 0]
    y = p[:, 1]
    return  (x - cxy[0])**2 + (y - cxy[1])**2 - r**2

def sign(phi):
    eps = 1e-12
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

    isXEdge = np.zeros(NE, dtype=np.bool)
    isXEdge[ny*(nx+1):] = True
    
    dx0[edge[isXEdge, 0]] = (phi[edge[isXEdge, 1]] - phi[edge[isXEdge, 0]])/dx
    dx1[edge[isXEdge, 1]] = dx0[edge[isXEdge, 0]] 

    dy0[edge[~isXEdge, 0]] = (phi[edge[~isXEdge, 1]] - phi[edge[~isXEdge, 0]])/dy
    dy1[edge[~isXEdge, 1]] = dy0[edge[~isXEdge, 0]]

    dx0[-ny-1:] = dx0[-2*(ny+1):-ny-1]
    dx1[0:ny+1] = dx1[ny+1:2*(ny+1)]
    dy0[ny::ny+1] = dy0[ny-1:-2:ny+1]
    dy1[0:-ny-1:ny+1] = dy1[1:-ny:ny+1]

    dxx = (dx0 - dx1)/dx
    dyy = (dy0 - dyz)/dx

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

    hg = np.zeros(N, dtype=np.float)

    hg[phiSign>=0] = np.sqrt(np.maximum(np.minimum(a[phiSign>=0], 0)**2, np.maximum(b[phiSign>=0], 0)**2) 
            + np.maximum(np.minimum(c[phiSign>=0], 0)**2, np.maximum(d[phiSign>=0], 0)**2))
    hg[phiSign < 0] = np.sqrt(np.maximum(np.maximum(a[phiSign<0], 0)**2, np.minimum(b[phiSign<0], 0)**2) 
            + np.maximum(np.maximum(c[phiSign<0], 0)**2, np.minimum(d[phiSign<0], 0)**2))

    return hg


box = [-1, 1, -1, 1]
cxy = (0.0, 0.0)
r = 0.5
interface = lambda p: dcircle(p, cxy, r)

nx = 10
ny = 10

mesh = StructureQuadMesh(box, nx, ny)

phi = interface(mesh.point)

phiSign = sign(phi)
edge = mesh.ds.edge
isCutEdge = phiSign[edge[:, 0]]*phiSign[edge[:, 1]] < 0
A = point[edge[isCutEdge, 0]]
B = point[edge[isCutEdge, 1]]
cutPoint = find_cut_point(self.interface, A, B)
h = np.sqrt(np.sum((cutPoint - A)**2, axis=1))


