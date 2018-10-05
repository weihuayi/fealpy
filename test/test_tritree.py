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
            NC = self.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool)
            isMarkedCell[idx] = True

            isTwoChildCell = (self.child[:, 1] > -1 & self.child[:, 2] == -1)
            idx0, = np.nonzero(isTwoChildCell)
            flag0 = np.ones(NC, dtype=np.bool)
            flag0[child[idx0, [0, 1]]] = True

            # expand the marked cell
            isExpand = np.zeros(NC, dtype=np.bool)
            cell2cell = self.ds.cell_to_cell()
            flag1 = (~isMarkedCell) & (~flag0) & (np.sum(isMarkedCell[cell2cell], axis=1) > 1)
            flag2 = (~isMarkedCell) & flag0 & (np.sum(isMarkedCell[cell2cell], axis=1) > 0)
            flag = flag1 | flag2
            while np.any(flag):
                isMarkedCell[flag] = True
                flag1 = (~isMarkedCell) & (~flag0) & (np.sum(isMarkedCell[cell2cell], axis=1) > 1)
                flag2 = (~isMarkedCell) & flag0 & (np.sum(isMarkedCell[cell2cell], axis=1) > 0)
                flag = flag1 | flag2

            if len(idx0) > 0:
                # delete the children of the cells with two children
                flag = isMarkedCell[child[idx0, 0]] | isMarkedCell[child[idx0, 1]]
                isMarkCell[idx0[flag]] = True
                self.child[idx0, 0:1] = -1

                flag = np.ones(NC, dtype=np.bool)
                flag[child[idx0, [0, 1]]] = False
                NN = self.number_of_nodes()
                self.ds.reinit(NN, cell[flag])
                self.parent = self.parent[flag]
                self.child = self.child[flag]
                isMarkedCell = isMarkedCell[flag]

            
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()
            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')


            # Find the cutted edge  
            cell2edge = self.ds.cell_to_edge()

            isCutEdge = np.zeros(NE, dtype=np.bool)
            isCutEdge[cell2edge[isMarkedCell, :]] = True

            isLeafCell = self.is_leaf_cell()
            isCuttedEdge = np.zeros(NE, dtype=np.bool)
            isCuttedEdge[cell2edge[~isLeafCell, :]] = True

            isCuttedEdge = isCuttedEdge & isCutEdge
            isNeedCutEdge = (~isCuttedEdge) & isCutEdge 

            # 找到每条非叶子边对应的单元编号， 及在该单元中的局部编号 

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
