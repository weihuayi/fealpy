import numpy as np
import  matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

from fealpy.mesh import TetrahedronMesh
from fealpy.mesh.mesh_quality import RadiusRatioQuality
from app.HVSCMesh.optimizer import *

def BlockJacobi3d(node,A,B0,B1,B2,isFreeNode):
    NN = node.shape[0]
    isBdNode = ~isFreeNode
    newNode = np.zeros((NN, 3), dtype=np.float64)
    newNode[isBdNode, :] = node[isBdNode, :]
    b = -B2*node[:, 1] -B1*node[:, 2] - A*newNode[:, 0]
    newNode[isFreeNode, 0], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
            b[isFreeNode], x0=node[isFreeNode, 0], tol=1e-6)
    b = B2*node[:, 0] - A*newNode[:, 1] - B0*node[:, 2]
    newNode[isFreeNode, 1], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
            b[isFreeNode], x0=node[isFreeNode, 1], tol=1e-6)
    b = B1*node[:,0]+B0*node[:,1]-A*newNode[:,2]
    newNode[isFreeNode, 2], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
            b[isFreeNode], x0=node[isFreeNode, 2], tol=1e-6)
    #node[isFreeNode, :] = newNode[isFreeNode, :]
    p = np.zeros((NN,3),dtype=np.float64)
    p[isFreeNode,:] = newNode[isFreeNode,:] - node[isFreeNode,:]
    #node += 0.7*p
    node += 0.3*p
    return node

node = np.array([
    [0.0,0.0,0.0],
    [0.5,0.0,0.0],
    [0.25,np.sqrt(3)/4,0.0],
    [1.0,0.25,0.005]],dtype=np.float64)
cell = np.array([[0, 1, 2, 3]],dtype=np.int64)

mesh = TetrahedronMesh(node, cell)
node = mesh.entity('node')
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

meshquality = RadiusRatioQuality(mesh)
isFreeNode = np.array([0,0,1,1],dtype=np.bool)
volume = mesh.entity_measure('cell')

q = np.zeros((2, NC),dtype=np.float64)
q[0] = meshquality(node)
minq = np.min(q[0])
avgq = np.mean(q[0])
maxq = np.max(q[0])
print('iter=',0,'minq=',minq,'avgq=',avgq, 'maxq=',maxq)

for i in range(0,100):
    A,B0,B1,B2 = meshquality.hess(node)
    mu = meshquality(node)
    cm = mesh.entity_measure("cell")
    
    A *= cm**2
    B0 *= cm**2
    B1 *= cm**2
    B2 *= cm**2
    
    I = np.broadcast_to(cell[:,:,None],(NC,4,4))
    J = np.broadcast_to(cell[:,None,:],(NC,4,4))
    A = csr_matrix((A.flat,(I.flat,J.flat)),shape=(NN,NN))
    B0 = csr_matrix((B0.flat, (I.flat, J.flat)), shape=(NN, NN))
    B1 = csr_matrix((B1.flat, (I.flat, J.flat)), shape=(NN, NN))
    B2 = csr_matrix((B2.flat, (I.flat, J.flat)), shape=(NN, NN))
    node = BlockJacobi3d(node,A,B0,B1,B2,isFreeNode)
    fname = './datatet/tet'+str(i)+'.vtu'
    mesh.to_vtk(fname)
    q[1] = meshquality(node)
    minq = np.min(q[1])
    avgq = np.mean(q[1])
    maxq = np.max(q[1])
    print('iter=',i+1,'minq=',minq,'avgq=',avgq,'maxq=',maxq)
    if np.max(np.abs(q[1]-q[0]))<1e-8:
        print("Bjacobi迭代次数为%d次"%(i+1))
        break
    q[0]=q[1]
mesh = TetrahedronMesh(node,cell)

