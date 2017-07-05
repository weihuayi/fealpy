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

def HG(mesh, phi):

    N = mesh.number_of_points()
    NE = mesh.number_of_edges()
    nx = mesh.ds.nx 
    ny = mesh.ds.ny
    dx = mesh.dx
    dy = mesh.dy

    phiSign = sign(phi)

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
    A = point[edge[isCutEdge, 0]]
    B = point[edge[isCutEdge, 1]]
    cutPoint = find_cut_point(self.interface, A, B)
    h = np.sqrt(np.sum((cutPoint - A)**2, axis=1))
    dx0[edge[isXEdge & isCutEdge, 0]] = -phi[edge[isXEdge & isCutEdge, 0]]/h[isXEdge[isCutEdge]]
    dx1[edge[isXEdge & isCutEdge, 1]] = phi[edge[isXEdge & isCutEdge, 1]]/(dx - h[isXEdge[isCutEdge]])
    dy0[edge[~isXEdge & isCutEdge, 0]] = -phi[edge[~isXEdge & isCutEdge, 0]]/h[~isXEdge[isCutEdge]]
    dy1[edge[~isXEdge & isCutEdge, 1]] = phi[edge[~isXEdge & isCutEdge, 1]]/(dy - h[~isXEdge[isCutEdge]])






#
#    Phi = phi.reshape(nx+1, ny+1) 
#    dx = mesh.dx
#    dy = mesh.dy
#
#    dx0 = np.zeros(nx+1, ny+1)
#    dx1 = np.zeros(nx+1, ny+1)
#    
#    dx0[0:-1, :] = (Phi[1:, :] - Phi[0:-1, :])/dx
#    dx1[1:, :] = (Phi[1:, :] - Phi[0:-1, :])/dx
#    dx0[-1, :] = dx1[-1, :]
#    dx1[0, :] = dx0[0, :]
#
#    dxx = (dx0 - dx1)/dx
#
#    dy0 = np.zeros(nx+1, ny+1)
#    dy1 = np.zeros(nx+1, ny+1)
#
#    dy0[:, 0:-1] = (Phi[:, 1:] - Phi[:, 0:-1])/dy
#    dy1[:, 1:] = (Phi[:, 1:] - Phi[:, 0:-1])/dy
#    dy0[:, -1] = dy1[:, -1]
#    dy1[:, 0] = dy0[:, 0]
#
#    dyy = (dy0 - dy1)/dy
#
#    minmodx = np.minimum(np.abs(dxx[0:-1, :]),  np.abs(dxx[1:, :]))
#    flag = dxx[0:-1, :]*dxx[1:, :] > 0
#    dx0[0:-1, :][flag] -= dx/2*minmodx[flag]
#    dx1[1:, :][flag] += dx/2*minmodx[flag]
#
#    minmody = np.minimum(np.abs(dyy[:, 0:-1]),   np.abs(dyy[:, 1:]))
#    flag = dyy[:, 0:-1]*dyy[:, 1:] > 0
#    dy0[:, 0:-1][flag] -= dy/2*minmody[flag]
#    dy1[:, 1:][flag] += dy/2*minmody[flag]
#
#    N = mesh.number_of_points()
#    NN = np.arange(N).reshape(nx+1, ny+1)




 
   
    

box = [-1, 1, -1, 1]
cxy = (0.0, 0.0)
r = 0.5
interface = lambda p: dcircle(p, cxy, r)

nx = 10
ny = 10

mesh = StructureQuadMesh(box, nx, ny)

phi = interface(mesh.point)



