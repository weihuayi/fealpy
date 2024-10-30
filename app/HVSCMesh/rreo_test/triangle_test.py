import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import csr_matrix

from fealpy.mesh import TriangleMesh
from fealpy.mesh.mesh_quality import RadiusRatioQuality
from app.HVSCMesh.optimizer import *

def BlockJacobi2d(node,A,B,isFreeNode):
    NN = node.shape[0] 
    isBdNode = ~isFreeNode
    newNode = np.zeros((NN, 2), dtype=bm.float64)
    newNode[isBdNode, :] = node[isBdNode, :]        
    b = -B*node[:, 1] - A*newNode[:, 0]
    newNode[isFreeNode, 0], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
            b[isFreeNode], x0=node[isFreeNode, 0], tol=1e-6)
    b = B*node[:, 0] - A*newNode[:, 1]
    newNode[isFreeNode, 1], info = cg(A[np.ix_(isFreeNode, isFreeNode)],
            b[isFreeNode], x0=node[isFreeNode, 1], tol=1e-6)
    node[isFreeNode, :] = newNode[isFreeNode, :]
    return node


node = np.array([[0.0,0.0],[2.0,0.0],[1,np.sqrt(3)]],dtype=np.float64)
cell = np.array([[0,1,2]],dtype=np.int64)
mesh = TriangleMesh(node,cell)
mesh.uniform_refine(2)
node = mesh.entity('node')
cell = mesh.entity('cell')

node[cell[-1,0]] = (node[cell[-1,0]]+node[cell[-1,2]])/2
node[cell[-1,1]] = (node[cell[-1,1]]+node[cell[-1,2]])/2
node[cell[-1,2],1] = 0.65

mesh.uniform_refine(3)
node = mesh.entity('node')
cell = mesh.entity('cell')
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
isBdNode = mesh.boundary_node_flag()
isFreeNode = ~isBdNode

meshquality = RadiusRatioQuality(mesh)

q = np.zeros((2, NC),dtype=np.float64)
q[0] = meshquality(node)
minq = np.min(q[0])
avgq = np.mean(q[0])
maxq = np.max(q[0])
print('iter=',0,'minq=',minq,'avgq=',avgq, 'maxq=',maxq)

for i in range(0,100):
    A,B = meshquality.hess(node)
    mu = meshquality(node)
    
    area = mesh.entity_measure('cell')
    area3 = area**3
    A *= area3[:,None,None]
    B *= area3[:,None,None]
    
    I = np.broadcast_to(cell[:,:,None],(NC,3,3))
    J = np.broadcast_to(cell[:,None,:],(NC,3,3))
    A = csr_matrix((A.flat,(I.flat,J.flat)),shape=(NN,NN))
    B = csr_matrix((B.flat,(I.flat,J.flat)),shape=(NN,NN))    
    node = BlockJacobi2d(node,A,B,isFreeNode)
    fname = 'datatri/tri'+str(i)+'.vtu'
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
mesh = TriangleMesh(node,cell)






