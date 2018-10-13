import numpy as np
from fealpy.mesh import TriangleMesh 

class Tritree(TriangleMesh):
    def __init__(self, node, cell):
        super(Tritree, self).__init__(node, cell)
        NC = self.number_of_cells()
        self.parent = -np.ones((NC, 2), dtype=self.itype)
        self.child = -np.ones((NC, 4), dtype=self.itype)
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

    def refine(self, idx):
        if len(idx) > 0:
            # Prepare data
            NC = self.number_of_cells()
            isMarkedCell = np.zeros(NC, dtype=np.bool)
            isMarkedCell[idx] = True
        
            isTwoChildCell = (self.child[:, 1] > -1) & (self.child[:, 2] == -1)
            flag0 = np.zeros(NC, dtype=np.bool)
            idx0, = np.nonzero(isTwoChildCell)
            if len(idx0) > 0:
                flag0[self.child[idx0, [0, 1]]] = True
        
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
                  
            cell2edge = self.ds.cell_to_edge()
       
            isCutEdge = np.zeros(NE, dtype=np.bool)
            isCutEdge[cell2edge[isMarkedCell, :]] = True

            isLeafCell = self.is_leaf_cell()
            isCuttedEdge = np.zeros(NE, dtype=np.bool)
            isCuttedEdge[cell2edge[~isLeafCell, :]] = True
            isCuttedEdge = isCuttedEdge & isCutEdge
           
            isNeedCutEdge = (~isCuttedEdge) & isCutEdge 
       
            edge2center = np.zeros(NE, dtype=np.int) 

            edge2cell = self.ds.edge_to_cell()
            flag = isLeafCell[edge2cell[:, 0]] & (~isLeafCell[edge2cell[:, 1]]) 
            I = self.child[edge2cell[flag, 1], 3]
            J = edge2cell[flag, 3]
            edge2center[flag] = cell[I, J]

            flag = (~isLeafCell[edge2cell[:, 0]]) & isLeafCell[edge2cell[:, 1]]
            I = self.child[edge2cell[flag, 0], 3]
            J = edge2cell[flag, 2]
            edge2center[flag] = cell[I, J]

            ec = self.entity_barycenter('edge', isNeedCutEdge) 
            NEC = len(ec)
            edge2center[isNeedCutEdge] = np.arange(NN, NN+NEC)
                
                                                                              
            # 一分为四的单元
            cell4Idx, = np.where(isMarkedCell)
            NCC = np.sum(isMarkedCell)
            cell4 = np.zeros((4*NCC, 3), dtype=self.itype)
            child4 = -np.ones((4*NCC, 4), dtype=self.itype)
            parent4 = -np.ones((4*NCC,2), dtype=self.itype) 
            cell4[:NCC, 0] = cell[isMarkedCell, 0] 
            cell4[:NCC, 1] = edge2center[cell2edge[isMarkedCell, 2]] 
            cell4[:NCC, 2] = edge2center[cell2edge[isMarkedCell, 1]] 
            parent4[:NCC, 0] = cell4Idx
            parent4[:NCC, 1] = 0

            cell4[NCC:2*NCC, 0] = cell[isMarkedCell, 1] 
            cell4[NCC:2*NCC, 1] = edge2center[cell2edge[isMarkedCell, 0]] 
            cell4[NCC:2*NCC, 2] = edge2center[cell2edge[isMarkedCell, 2]] 
            parent4[NCC:2*NCC, 0] = cell4Idx
            parent4[NCC:2*NCC, 1] = 1

            cell4[2*NCC:3*NCC, 0] = cell[isMarkedCell, 2] 
            cell4[2*NCC:3*NCC, 1] = edge2center[cell2edge[isMarkedCell, 1]] 
            cell4[2*NCC:3*NCC, 2] = edge2center[cell2edge[isMarkedCell, 0]] 
            parent4[2*NCC:3*NCC, 0] = cell4Idx
            parent4[2*NCC:3*NCC, 1] = 2

            cell4[3*NCC:4*NCC, 0] = edge2center[cell2edge[isMarkedCell, 0]] 
            cell4[3*NCC:4*NCC, 1] = edge2center[cell2edge[isMarkedCell, 1]] 
            cell4[3*NCC:4*NCC, 2] = edge2center[cell2edge[isMarkedCell, 2]]
            parent4[3*NCC:4*NCC, 0] = cell4Idx
            parent4[3*NCC:4*NCC, 1] = 3

            # 一分为二的单元
            cell2Idx = np.where(isTwoChildCell)
            N2CC = np.sum(isTwoChildCell)
            cell2 = np.zeros((2*N2CC, 3), dtype=self.itype)
            child4 = -np.ones((2*N2CC, 4), dtype=self.itype)
            parent2 = -np.ones((2*N2CC, 4), dtype=self.itype)
            cell2[0:N2CC, 0] = cell[isTwoChildCell, 0]
            cell2[0:N2CC, 1] = cell[isTwoChildCell, 1]
            cell2[0:N2CC, 2] = edge2center[cell2edge[cell[I, J], 1]]
            parent2[0:N2CC, 0] = cell2Idx
            parent2[0:N2CC, 1] = 0

            cell2[N2CC:2*N2CC, 0] = cell[isTwoChildCell, 1]                           
            cell2[N2CC:2*N2CC, 1] = cell[isTwoChildCell, 2]                           
            cell2[N2CC:2*N2CC, 2] = edge2center[cell2edge[isTwoChildCell, 1]]          
            parent2[N2CC:2*N2CC, 0] = cell2Idx                                        
            parent2[N2CC:2*N2CC, 1] = 1
            
#            child[idx, :] = np.arange(NC, NC + 4*NCC).reshape(NCC, 4)           
#                                                                                             
            cell = np.concatenate((cell, cell4, cell2), axis=0)                      
            self.node = np.concatenate((node, ec), axis=0)  
#            self.parent = np.concatenate((parent, parent4, parent2), axis=0)           
#            self.child = np.concatenate((child, child4, child2), axis=0)              
            self.ds.reinit(NN + NEC + NCC + N2CC, cell) 
               
          













































#mesh_info = MeshInfo()
#mesh_info.set_points([(-1,-1),(0,-1),(0,0),(1,0),(1,1),(0,1),(1,1),(-1,0)])
#mesh_info.set_facets([[0,1],[1,2],[2,3],[3,4],[4, 5],[5,6],[6,7],[7,0]]) 
#
#h = 0.05
#mesh = build(mesh_info, max_volume=h**2)
#node = np.array(mesh.points, dtype=np.float)
#cell = np.array(mesh.elements, dtype=np.int)
#ttree = Tritree(node, cell)
#mesh = ttree.to_mesh()
#
#pde = LShapeRSinData()
#integrator = mesh.integrator(3)
#fem = PoissonFEMModel(pde, mesh, 1, integrator)
#fem.solve()
#eta = fem.recover_estimate()
#ttree.refine(marker=AdaptiveMarker(eta, theta=theta))
#
#fig = plt.figure()
#axes = fig.gca()
#mesh.add_plot(axes, cellcolor='w', markersize=200)
#plt.show()
