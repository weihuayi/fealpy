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
    ny = mesh.dx.ny

    phiSign = sign(phi)

    dx0 = np.zeros(N, dtype=np.float)
    dx1 = np.zeros(N, dtype=np.float)
    dy0 = np.zeros(N, dtype=np.float)
    dy1 = np.zeros(N, dtype=np.float)

    edge = mesh.ds.edge
    isCutEdge = phiSign[edge[:, 0]]*phiSign[edge[:, 1]] < 0

    A = point[edge[isCutEdge, 0]]
    B = point[edge[isCutEdge, 1]]

    cutPoint = find_cut_point(self.interface, A, B)
    

    isXEdge = np.zeros(NE, dtype=np.bool)
    isXEdge[ny*(nx+1):] = True

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



