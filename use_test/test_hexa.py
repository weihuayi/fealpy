from fealpy.mesh import HexahedronMesh
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mesh = HexahedronMesh.from_one_hexahedron()
node = mesh.entity('node')
edge = mesh.entity('edge')
face = mesh.entity('face')
cell = mesh.entity('cell')
print(node)
print(edge)
print(face)
print(cell)

# def Prolongation_matrix(mesh,numnode,numedge,numface,numcell):
#     new_node_num = numnode+numedge+numface+numcell
#     nonzeros = numnode+2*numedge+4*numface+8*numcell
#     data = np.zeros(nonzeros,dtype=np.float64)
#     data[:numnode] = 1
#     data[numnode:numnode+2*numedge] = 1/2
#     data[2*numedge:numnode+2*numedge+4*numface] = 1/4
#     data[numnode+2*numedge+4*numface:] = 1/8

#     indices = np.zeros(nonzeros,dtype=np.int32)
#     indices[:numnode] = np.arange(numnode)
#     indices[numnode:numnode+2*numedge] = edge2node.flatten()
#     indices[numnode+2*numedge:numnode+2*numedge+4*numface] = face2node.flatten()
#     indices[numnode+2*numedge+4*numface:] = cell2node.flatten()

#     indptr = np.zeros(new_node_num+1,dtype=np.int32)
#     indptr[:numnode+1] = np.arange(numnode+1)
#     indptr[numnode+1:numnode+numedge+1]=np.arange(numnode+2,numnode+2*numedge+1,step=2)
#     indptr[numnode+numedge+1:numnode+numedge+numface+1] = np.arange(numnode+2*numedge+4,numnode+2*numedge+4*numface+1,step=4)
#     indptr[numnode+numedge+numface+1:] = np.arange(numnode+2*numedge+4*numface+8,nonzeros+1,step=8)

#     A = csr_matrix((data,indices,indptr),dtype=np.float64)
#     return A

# for i in np.arange(n):
#     '''
#     n:加密次数
#     '''
#     edge2node = mesh.edge_to_node()
#     face2node = mesh.face_to_node()
#     cell2node = mesh.cell_to_node()
#     NN = mesh.number_of_nodes()
#     NE = mesh.number_of_edges()
#     NF = mesh.number_of_faces()
#     NC= mesh.number_of_cells()
#     mesh.uniform_refine()

#     A = Prolongation_matrix(mesh,NN,NE,NF,NC)
#     IM.append(A)

# print(IM[1])

n = 2
A = mesh.uniform_refine(n,returnim=True)

def linear_function(coodrs):
    return 2*coodrs[:,0] + 2*coodrs[:,1]+2*coodrs[:,2]+1
excat_value = linear_function(mesh.node)
val = linear_function(node)
test_value = val
for i in range(n):
    print(A[i])
    test_value = A[i].dot(test_value)
print(excat_value)
print(test_value)

# fig = plt.figure(1)
# axes = fig.add_subplot(111, projection='3d')
# mesh.add_plot(axes)
# plt.title("Grid Image")
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# plt.show()