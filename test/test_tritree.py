import numpy as np
from fealpy.mesh import TriangleMesh 
from meshpy.triangle import MeshInfo, build
import matplotlib.pyplot as plt 

from fealpy.model.poisson_model_2d import LShapeRSinData 

from fealpy.femmodel.PoissonFEMModel import PoissonFEMModel
from fealpy.tools.show import showmultirate
from fealpy.mesh.adaptive_tools import AdaptiveMarker 

class Tritree(TriangleMesh):
    def __init__(self, node, cell):
        super(Tritree, self).__init__(node, cell)
        NC = self.number_of_cells()
        self.parent = -np.ones((NC, 2), dtype=np.float)
        self.child = -np.ones((NC, 4), dtype=np.float)
        self.meshtype = 'tritree'

    def leaf_cell_index(self):
        child = self.child
        idx, = np.nonzero(child[:, 0] == -1)
        return idx

    def leaf_cell(self):
        child = self.child
        cell = self.ds.cell[child[:, 0] == -1]
        return cell

    def is_leaf_cell(self, idx=None):
        if idx is None:
            return self.child[:, 0] == -1
        else:
            return self.child[idx, 0] == -1

    def is_root_cell(self, idx=None):
        if idx is None:
            return self.parent[:, 0] == -1
        else:
            return self.parent[idx, 0] == -1

    def to_mesh(self):
        isLeafCell = self.is_leaf_cell()
        return TriangleMesh(self.node, self.ds.cell[isLeafCell])

    def refine(self, marker=None):
        if marker == None:
            idx = self.leaf_cell_index()
        else:
            # 需要加密的叶子单元编号
            idx = marker.refine_marker(self)

        if idx is None:
            return False

        if len(idx) > 0:
            # Prepare data
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')

            parent = self.parent
            child = self.child
            isLeafCell = self.is_leaf_cell()

            # Construct 
            isNeedCutCell = np.zeros(NC, dtype=np.bool)
            isNeedCutCell[idx] = True
            isNeedCutCell = isNeedCutCell & isLeafCell



            # Find the cutted edge  
            cell2edge = self.ds.cell_to_edge()

            isCutEdge = np.zeros(NE, dtype=np.bool)
            isCutEdge[cell2edge[isNeedCutCell, :]] = True

            isCuttedEdge = np.zeros(NE, dtype=np.bool)
            isCuttedEdge[cell2edge[~isLeafCell, :]] = True
            isCuttedEdge = isCuttedEdge & isCutEdge

            isNeedCutEdge = (~isCuttedEdge) & isCutEdge 

            # 找到每条非叶子边对应的单元编号， 及在该单元中的局部编号 
            I, J = np.nonzero(isCuttedEdge[cell2edge])
            cellIdx = np.zeros(NE, dtype=np.int)
            localIdx = np.zeros(NE, dtype=np.int)
            I1 = I[~isLeafCell[I]]
            J1 = J[~isLeafCell[I]]
            cellIdx[cell2edge[I1, J1]] = I1 # the cell idx 
            localIdx[cell2edge[I1, J1]] = J1 #
            del I, J, I1, J1

            # 找到该单元相应孩子单元编号， 及对应的中点编号
            cellIdx = cellIdx[isCuttedEdge]
            localIdx = localIdx[isCuttedEdge]
            cellIdx = child[cellIdx, self.localEdge2childCell[localIdx, 0]]
            localIdx = self.localEdge2childCell[localIdx, 1]

            edge2center = np.zeros(NE, dtype=np.int)
            edge2center[isCuttedEdge] = cell[cellIdx, localIdx]  

            edgeCenter = 0.5*np.sum(node[edge[isNeedCutEdge]], axis=1) 
            cellCenter = self.entity_barycenter('cell', isNeedCutCell)

            NEC = len(edgeCenter)
            NCC = len(cellCenter)

            edge2center[isNeedCutEdge] = np.arange(N, N+NEC) 

            cp = [cell[isNeedCutCell, i].reshape(-1, 1) for i in range(4)]
            ep = [edge2center[cell2edge[isNeedCutCell, i]].reshape(-1, 1) for i in range(4)]
            cc = np.arange(N + NEC, N + NEC + NCC).reshape(-1, 1)
            
            newCell = np.zeros((4*NCC, 4), dtype=np.int)
            newChild = -np.ones((4*NCC, 4), dtype=np.int)
            newParent = -np.ones((4*NCC, 2), dtype=np.int)
            newCell[0::4, :] = np.concatenate((cp[0], ep[0], cc, ep[3]), axis=1) 
            newCell[1::4, :] = np.concatenate((ep[0], cp[1], ep[1], cc), axis=1)
            newCell[2::4, :] = np.concatenate((cc, ep[1], cp[2], ep[2]), axis=1)
            newCell[3::4, :] = np.concatenate((ep[3], cc, ep[2], cp[3]), axis=1)
            newParent[:, 0] = np.repeat(idx, 4)
            newParent[:, 1] = ranges(4*np.ones(NCC, dtype=np.int)) 
            child[idx, :] = np.arange(NC, NC + 4*NCC).reshape(NCC, 4)

            cell = np.concatenate((cell, newCell), axis=0)
            self.node = np.concatenate((node, edgeCenter, cellCenter), axis=0)
            self.parent = np.concatenate((parent, newParent), axis=0)
            self.child = np.concatenate((child, newChild), axis=0)
            self.ds.reinit(N + NEC + NCC, cell)
            return True
        else:
            return False


mesh_info = MeshInfo()
mesh_info.set_points([(-1,-1),(0,-1),(0,0),(1,0),(1,1),(0,1),(1,1),(-1,0)])
mesh_info.set_facets([[0,1],[1,2],[2,3],[3,4],[4, 5],[5,6],[6,7],[7,0]]) 

h = 0.05
mesh = build(mesh_info, max_volume=h**2)
node = np.array(mesh.points, dtype=np.float)
cell = np.array(mesh.elements, dtype=np.int)
ttree = Tritree(node, cell)
mesh = ttree.to_mesh()

pde = LShapeRSinData()
integrator = mesh.integrator(3)
fem = PoissonFEMModel(pde, mesh, 1, integrator)
fem.solve()
eta = fem.recover_estimate()
ttree.refine(marker=AdaptiveMarker(eta, theta=theta))

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes, cellcolor='w', markersize=200)
plt.show()
