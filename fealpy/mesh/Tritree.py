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

    def refineRG(self, marker=None):
        if marker == None:
            idx = self.leaf_cell_index()
        else:
            idx = marker.refine_marker(self)

        if idx is None:
            return False

        if len(idx) > 0:

            NN = self.number_of_nodes()
            NE = self.number_of_edges()

            # Prepare data

            # Prepare data
            NN = self.number_of_nodes()
            NE = self.number_of_edges()

            NC = self.number_of_cells()

            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')
            

            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')

            isMarkedCell = np.zeros(NC, dtype=np.bool)
            isMarkedCell[idx] = True

            # find the element with two children

            isTwoChildCell = (self.child[:, 1] > -1) & (self.child[:, 2] == -1)
            flag0 = np.zeros(NC, dtype=np.bool)

            isTwoChildCell = (self.child[:, 1] > -1) & (self.child[:, 2] == -1)

            flag0 = np.zeros(NC, dtype=np.bool)
            isTwoChildCell = (self.child[:, 2] > -1) & (self.child[:, 3] == -1)
            isFourChildCell = self.child[:, 3] > -1

            flag0[self.child[isTwoChildCell, 0:2]] = True

            
            # expand the marked cell

            # expand the marked cell


            # expand the marked cell. 
            assert self.irule == 1  # TODO: add support for  general k irregular rule case 

            cell2cell = self.ds.cell_to_cell()
            flag = (~isMarkedCell) & (np.sum(isMarkedCell[cell2cell], axis=1) > 1)
            while np.any(flag):
                isMarkedCell[flag] = True

                flag1 = (~isMarkedCell) & (~flag0) & (np.sum(isMarkedCell[cell2cell], axis=1) > 1)
                flag2 = (~isMarkedCell) & flag0 & (np.sum(isMarkedCell[cell2cell], axis=1) > 0)
                flag = flag1 | flag2
            
                flag1 = (~isMarkedCell) & (~flag0) & (np.sum(isMarkedCell[cell2cell], axis=1) > 1)
                flag2 = (~isMarkedCell) & flag0 & (np.sum(isMarkedCell[cell2cell], axis=1) > 0)
                flag = flag1 | flag2

                flag = (~isMarkedCell) & (np.sum(isMarkedCell[cell2cell], axis=1) > 1)



            cell2edge = self.ds.cell_to_edge()

            edge2Newnode = np.zeros(NE, dtype=np.bool)
            edge2Newnode[cell2edge[isMarkedCell]] = True
                  
            cell = self.entity("cell")

            NE = self.number_of_edges()
            edge2newNode = np.zeros(NE, dtype=np.bool)
            edge2newNode[cell2edge[isMarkedCell]] = True
        

            refineFlag = np.zeros(NE, dtype=np.bool)
            refineFlag[cell2edge[isMarkedCell]] = True
        
            edge2cell = self.ds.edge_to_cell()
            isBdEdge = self.ds.boundary_edge_flag()
            flag = flag0[edge2cell[:, 0]] & flag0[edge2cell[:, 1]] & (~isBdEdge)

            edge2Newnode[flag] = False
            edge2Newnode[cell]

            edge2newNode[flag] = False
            edge2newNode[cell]

    def refine(self, marker=None):
        if marker == None:
            idx = self.leaf_cell_index()
        else:
            idx = marker.refine_marker(self)

            refineFlag[flag] = False

            NNN = refineFlag.sum()
            edge2newNode = np.zeros(NE, dtype=self.itype)
            edge2newNode[refineFlag] = NN + range(NNN)

            greenIdx, = np.where(isTwoChildCell)
            edge2newNode[cell2edge[greenIdx, self.child[greenIdx, 2]]] = cell[self.child[greenIdx, 0], 0]

            # red cell 
            redFlag =  ((~flag0) & isMarkedCell)
            redIdx, = np.where(redFlag)                                 
            NCC = len(redIdx) 
            cell4 = np.zeros((4*NCC, 3), dtype=self.itype)
            child4 = -np.ones((4*NCC, 4), dtype=self.itype)
            parent4 = -np.ones((4*NCC, 2), dtype=self.itype) 
            cell4[:NCC, 0] = cell[isMarkedCell, 0] 
            cell4[:NCC, 1] = edge2newNode[cell2edge[redFlag, 2]] 
            cell4[:NCC, 2] = edge2newNode[cell2edge[redFlag, 1]] 
            parent4[:NCC, 0] = redIdx
            parent4[:NCC, 1] = 0
            self.child[redIdx, 0] = NC + np.arange(0, NCC)

            cell4[NCC:2*NCC, 0] = cell[redFlag, 1] 
            cell4[NCC:2*NCC, 1] = edge2newNode[cell2edge[redFlag, 0]] 
            cell4[NCC:2*NCC, 2] = edge2newNode[cell2edge[redFlag, 2]] 
            parent4[NCC:2*NCC, 0] = redIdx
            parent4[NCC:2*NCC, 1] = 1
            self.child[redIdx, 1] = NC + np.arange(NCC, 2*NCC)

            cell4[2*NCC:3*NCC, 0] = cell[redFlag, 2] 
            cell4[2*NCC:3*NCC, 1] = edge2newNode[cell2edge[redFlag, 1]] 
            cell4[2*NCC:3*NCC, 2] = edge2newNode[cell2edge[redFlag, 0]] 
            parent4[2*NCC:3*NCC, 0] = redIdx
            parent4[2*NCC:3*NCC, 1] = 2
            self.child[redIdx, 2] = NC + np.arange(2*NCC, 3*NCC)

            cell4[3*NCC:4*NCC, 0] = edge2newNode[cell2edge[redFlag, 0]] 
            cell4[3*NCC:4*NCC, 1] = edge2newNode[cell2edge[redFlag, 1]] 
            cell4[3*NCC:4*NCC, 2] = edge2newNode[cell2edge[redFlag, 2]]
            parent4[3*NCC:4*NCC, 0] = redIdx
            parent4[3*NCC:4*NCC, 1] = 3
            self.child[redIdx, 3] = NC + np.arange(3*NCC, 4*NCC)


            





    def refine(self, marker=None):
        if marker == None:
            idx = self.leaf_cell_index()
        else:
            idx = marker.refine_marker(self)


            
            isCutEdge = np.zeros(NE, dtype=np.bool)
            isCutEdge[cell2edge[isMarkedCell, :]] = True
            
            isLeafCell = self.is_leaf_cell()
            isCuttedEdge = np.zeros(NE, dtype=np.bool)
            isCuttedEdge[cell2edge[~isLeafCell, :]] = True
            isCuttedEdge = isCuttedEdge & isCutEdge
            isNeedCutedEdge = (~isCuttedEdge) & isCutEdge

            flag0 = isLeafCell[edge2cell[:, 0]] & (~isLeafCell[edge2cell[:, 1]])
            I = self.child[edge2cell[flag0, 1], 3]
            J = edge2cell[flag0, 3]
            edge2center[flag0] = cell[I, J]

            flag1 = (~isLeafCell[edge2cell[:, 0]]) & isLeafCell[edge2cell[:, 1]]
            I = self.child[edge2cell[flag1, 0], 3]                              
            J = edge2cell[flag0, 2]                                              
            edge2center[flag1] = cell[I, J]
            
            edge2center = np.zeros(NE, dtype=np.int)
            ec = self.entity_barycenter('edge', isNeedCutEdge)
            NEC = len(ec)
            edge2center[isNeedCutEdge] = np.arange(NN, NN+NEC)

            #一分为四的单元
            cell4Idx, = np.where(isMarkedCell)
            NCC = np.sum(isMarkedCell)
            cell
            
                     
            
        
#    def refine(self, marker=None):
#        if marker == None:
#            idx = self.leaf_cell_index()
#        else:
#            idx = marker.refine_marker(self)
#
#        if idx is None:
#            return False
#
#        if len(idx) > 0:
#            # Prepare data
#            NC = self.number_of_cells()
#            isMarkedCell = np.zeros(NC, dtype=np.bool)
#            isMarkedCell[idx] = True
#        
#            isTwoChildCell = (self.child[:, 1] > -1) & (self.child[:, 2] == -1)
#            flag0 = np.zeros(NC, dtype=np.bool)
#            idx0, = np.nonzero(isTwoChildCell)
#            if len(idx0) > 0:
#                flag0[self.child[idx0, 0:2]] = True
#            
#            # expand the marked cell
#            cell2cell = self.ds.cell_to_cell()
#            flag1 = (~isMarkedCell) & (~flag0) & (np.sum(isMarkedCell[cell2cell], axis=1) > 1)
#            flag2 = (~isMarkedCell) & flag0 & (np.sum(isMarkedCell[cell2cell],axis=1) > 0)
#            flag = flag1 | flag2 
#            while np.any(flag):
#                isMarkedCell[flag] = True
#                flag1 = (~isMarkedCell) & (~flag0) & (np.sum(isMarkedCell[cell2cell], axis=1) > 1)
#                flag2 = (~isMarkedCell) & flag0 & (np.sum(isMarkedCell[cell2cell], axis=1) > 0)
#                flag = flag1 | flag2
#
#            if len(idx0) > 0:
#                # delete the children of the cells with two children
#                flag = isMarkedCell[self.child[idx0, 0]] | isMarkedCell[self.child[idx0, 1]]
#                
#                isMarkedCell[idx0[flag]] = True
#                flag = np.ones(NC, dtype=np.bool)
#                flag[self.child[idx0, 0:2]] = False           
#                NN = self.number_of_nodes()
#                cell = self.entity('cell')
#                self.ds.reinit(NN, cell[flag])
#
#                self.child[idx0, 0:2] = -1
#                self.parent = self.parent[flag]
#                self.child = self.child[flag]
#                isMarkedCell = isMarkedCell[flag]
#
#            NN = self.number_of_nodes()
#            NE = self.number_of_edges()
#            NC = self.number_of_cells()
#            node = self.entity('node')
#            edge = self.entity('edge')
#            cell = self.entity('cell')
#                  
#            cell2edge = self.ds.cell_to_edge()
#       
#            isCutEdge = np.zeros(NE, dtype=np.bool)
#            isCutEdge[cell2edge[isMarkedCell, :]] = True
#
#            isLeafCell = self.is_leaf_cell()
#            isCuttedEdge = np.zeros(NE, dtype=np.bool)
#            isCuttedEdge[cell2edge[~isLeafCell, :]] = True
#            isCuttedEdge = isCuttedEdge & isCutEdge
#           
#            isNeedCutEdge = (~isCuttedEdge) & isCutEdge 
#       
#            edge2center = np.zeros(NE, dtype=np.int) 
#            edge2cell = self.ds.edge_to_cell()
#
#            flag0 = isLeafCell[edge2cell[:, 0]] & (~isLeafCell[edge2cell[:, 1]]) 
#            I = self.child[edge2cell[flag0, 1], 3]
#            J = edge2cell[flag0, 3]
#            edge2center[flag0] = cell[I, J]
#
#            flag1 = (~isLeafCell[edge2cell[:, 0]]) & isLeafCell[edge2cell[:, 1]]
#            I = self.child[edge2cell[flag1, 0], 3]
#            J = edge2cell[flag1, 2]
#            edge2center[flag1] = cell[I, J]
#
#
#            ec = self.entity_barycenter('edge', isNeedCutEdge) 
#            NEC = len(ec)
#            edge2center[isNeedCutEdge] = np.arange(NN, NN+NEC)
#                
#                                                                              
#            # 一分为四的单元
#            cell4Idx, = np.where(isMarkedCell)                                 
#            NCC = np.sum(isMarkedCell)
#            cell4 = np.zeros((4*NCC, 3), dtype=self.itype)
#            child4 = -np.ones((4*NCC, 4), dtype=self.itype)
#            parent4 = -np.ones((4*NCC, 2), dtype=self.itype) 
#            cell4[:NCC, 0] = cell[isMarkedCell, 0] 
#            cell4[:NCC, 1] = edge2center[cell2edge[isMarkedCell, 2]] 
#            cell4[:NCC, 2] = edge2center[cell2edge[isMarkedCell, 1]] 
#            parent4[:NCC, 0] = cell4Idx
#            parent4[:NCC, 1] = 0
#            self.child[cell4Idx, 0] = NC + np.arange(0, NCC)
#
#            cell4[NCC:2*NCC, 0] = cell[isMarkedCell, 1] 
#            cell4[NCC:2*NCC, 1] = edge2center[cell2edge[isMarkedCell, 0]] 
#            cell4[NCC:2*NCC, 2] = edge2center[cell2edge[isMarkedCell, 2]] 
#            parent4[NCC:2*NCC, 0] = cell4Idx
#            parent4[NCC:2*NCC, 1] = 1
#            self.child[cell4Idx, 1] = NC + np.arange(NCC, 2*NCC)
#
#            cell4[2*NCC:3*NCC, 0] = cell[isMarkedCell, 2] 
#            cell4[2*NCC:3*NCC, 1] = edge2center[cell2edge[isMarkedCell, 1]] 
#            cell4[2*NCC:3*NCC, 2] = edge2center[cell2edge[isMarkedCell, 0]] 
#            parent4[2*NCC:3*NCC, 0] = cell4Idx
#            parent4[2*NCC:3*NCC, 1] = 2
#            self.child[cell4Idx, 2] = NC + np.arange(2*NCC, 3*NCC)
#
#            cell4[3*NCC:4*NCC, 0] = edge2center[cell2edge[isMarkedCell, 0]] 
#            cell4[3*NCC:4*NCC, 1] = edge2center[cell2edge[isMarkedCell, 1]] 
#            cell4[3*NCC:4*NCC, 2] = edge2center[cell2edge[isMarkedCell, 2]]
#            parent4[3*NCC:4*NCC, 0] = cell4Idx
#            parent4[3*NCC:4*NCC, 1] = 3
#            self.child[cell4Idx, 3] = NC + np.arange(3*NCC, 4*NCC)
#
#            # 一分为二的单元
#            flag0 = isLeafCell[edge2cell[:, 0]] & \
#                    (~isMarkedCell[edge2cell[:, 1]]) &\
#                    (isMarkedCell[edge2cell[:, 1]] | (~isLeafCell[edge2cell[:,1]]))
#            flag1 = isLeafCell[edge2cell[:, 1]] & \
#                    (~isMarkedCell[edge2cell[:, 0]]) & \
#                    (isMarkedCell[edge2cell[:, 0]] | (~isLeafCell[edge2cell[:,0]]))
#            
#            cidx0 = edge2cell[flag0, 0]
#            N0 = len(cidx0)
#            cell20 = np.zeros((2*N0, 3), dtype=self.itype)
#            child20 = -np.ones((2*N0, 4), dtype=self.itype)
#            parent20 = -np.ones((2*N0, 2), dtype=self.itype)
#
#            cell20[0:N0, 0] = edge[flag0, 0] 
#            cell20[0:N0, 1] = edge2center[flag0] 
#            cell20[0:N0, 2] = cell[cidx0, edge2cell[flag0, 2]] 
#            parent20[0:N0, 0] = cidx0 
#            parent20[0:N0, 1] = 0
#            self.child[cidx0, 0] = NC + 4*NCC + np.arange(0, N0)
#
#            cell20[N0:2*N0, 0] = edge[flag0, 1]                           
#            cell20[N0:2*N0, 1] = edge2center[flag0]                           
#            cell20[N0:2*N0, 2] = cell[cidx0, edge2cell[flag0, 2]]          
#            parent20[N0:2*N0, 0] = cidx0                                       
#            parent20[N0:2*N0, 1] = 1
#            self.child[cidx0, 1] = NC + 4*NCC + np.arange(N0, 2*N0)
#
#            cidx1 = edge2cell[flag1, 1]
#            N1 = len(cidx1)
#            cell21 = np.zeros((2*N1, 3), dtype=self.itype)
#            child21 = -np.ones((2*N1, 4), dtype=self.itype)
#            parent21 = -np.ones((2*N1, 2), dtype=self.itype)
#
#            cell21[0:N1, 0] = edge[flag1, 1] 
#            cell21[0:N1, 1] = edge2center[flag1] 
#            cell21[0:N1, 2] = cell[cidx1, edge2cell[flag1, 3]] 
#            parent21[0:N1, 0] = cidx1 
#            parent21[0:N1, 1] = 0
#            self.child[cidx1, 0] = NC + 4*NCC + 2*N0 + np.arange(0, N1)
#
#            cell21[N1:2*N1, 0] = edge[flag1, 0]                           
#            cell21[N1:2*N1, 1] = edge2center[flag1]                           
#            cell21[N1:2*N1, 2] = cell[cidx1, edge2cell[flag1, 3]]          
#            parent21[N1:2*N1, 0] = cidx1                                       
#            parent21[N1:2*N1, 1] = 1
#            self.child[cidx1, 1] = NC + 4*NCC + 2*N0 + np.arange(N1, 2*N1)
#
#            cell = np.r_['0', cell, cell4, cell20, cell21]                      
#            self.node = np.r_['0', node, ec] 
#            self.parent = np.r_['0', self.parent, parent4, parent20, parent21]           
#            self.child = np.r_['0', self.child, child4, child20, child21]              
#            self.ds.reinit(NN + NEC, cell) 

               
          













































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
