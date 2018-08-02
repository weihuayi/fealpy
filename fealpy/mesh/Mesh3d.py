import numpy as np

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
import mpl_toolkits.mplot3d as a3
from .mesh_tools import unique_row, find_entity, show_mesh_3d, find_node

class Mesh3d():
    def __init__(self):
        pass

    def number_of_nodes(self):
        return self.node.shape[0]

    def number_of_edges(self):
        return self.ds.NE

    def number_of_faces(self):
        return self.ds.NF

    def number_of_cells(self):
        return self.ds.NC

    def number_of_vertices_of_cells(self):
        return self.ds.number_of_vertices_of_cells()

    def number_of_edges_of_cells(self):
        return self.ds.number_of_edges_of_cells()

    def geo_dimension(self):
        return self.node.shape[1] 

    def top_dimension(self):
        return 3

    def entity(self, etype='cell'):
        if etype in ['cell', 3]:
            return self.ds.cell
        elif etype in ['face', 2]:
            return self.ds.face
        elif etype in ['edge', 1]:
            return self.ds.edge
        elif etype in ['node', 0]:
            return self.node
        else:
            raise ValueError("`entitytype` is wrong!")

    def entity_measure(self, etype=3):
        if etype in ['cell', 3]:
            return self.cell_volume()
        elif etype in ['face', 2]:
            return self.face_area()
        elif etype in ['edge', 1]:
            return self.edge_length()
        elif etype in ['node', 0]:
            NN = self.number_of_nodes()
            return np.zeros(NN, dtype=np.float)
        else:
            raise ValueError("`entitytype` is wrong!")

    def entity_barycenter(self, etype='cell'):
        node = self.node
        if etype in ['cell', 3]:
            cell = self.ds.cell
            bc = np.sum(node[cell, :], axis=1).reshape(-1, 3)/cell.shape[1]
        elif etype in ['face', 2]:
            face = self.ds.face
            bc = np.sum(node[face, :], axis=1).reshape(-1, 3)/face.shape[1]
        elif etype in ['edge', 1]:
            edge = self.ds.edge
            bc = np.sum(node[edge, :], axis=1).reshape(-1, 3)/edge.shape[1]
        elif etype in  ['node', 0]:
            bc = node
        else:
            raise ValueError("`etype` is wrong!")

        return bc

    def face_unit_normal(self):
        face = self.ds.face
        node = self.node
        v01 = node[face[:, 1], :] - node[face[:, 0], :]
        v02 = node[face[:, 2], :] - node[face[:, 0], :]
        dim = self.node.shape[1] 
        nv = np.cross(v01, v02)
        length = np.sqrt(np.square(nv).sum(axis=1))
        return nv/length.reshape(-1,1) 

    def edge_unit_tagent(self):
        edge = self.ds.edge
        node = self.node
        v = node[edge[:,1], :] - node[edge[:,0],:]
        length = np.sqrt(np.square(v).sum(axis=1))
        return v/length.reshape(-1,1)

    def add_plot(self, plot,
            nodecolor='k', edgecolor='k',
             aspect='equal',
            linewidths=2, markersize=20,  
            showaxis=False, alpha=0.8):

        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca() 
        else:
            axes = plot

        return show_mesh_3d(axes, self,
                nodecolor=nodecolor, edgecolor=edgecolor,
                aspect=aspect,
                linewidths=linewidths, markersize=markersize,  
                 showaxis=showaxis, alpha=alpha)

    def find_node(self, axes, node=None,
            index=None, showindex=False,
            color='r', markersize=40, 
            fontsize=24, fontcolor='k'):

        if node is None:
            node = self.node
        find_node(axes, node, 
                index=index, showindex=showindex, 
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    def find_edge(self, axes, 
            index=None, showindex=False,
            color='g', markersize=80, 
            fontsize=24, fontcolor='k'):

        find_entity(axes, self, entity='edge',
                index=index, showindex=showindex, 
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    def find_face(self, axes, 
            index=None, showindex=False,
            color='k', markersize=120, 
            fontsize=24, fontcolor='k'):
        find_entity(axes, self,  entity='face',
                index=index, showindex=showindex, 
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    def find_cell(self, axes, 
            index=None, showindex=False,
            color='r', markersize=20, 
            fontsize=24, fontcolor='k'):
        find_entity(axes, self, entity='cell',
                index=index, showindex=showindex, 
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)


class Mesh3dDataStructure():
    def __init__(self, NN, cell):

        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.construct()

    def reinit(self, NN, cell):
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.construct()

    def clear(self):
        self.face = None
        self.face2cell = None
        self.edge = None
        self.cell2edge = None

    def number_of_vertices_of_cells(self):
        return self.V

    def number_of_edges_of_cells(self):
        return self.E


    def total_edge(self):
        NC = self.NC
        cell = self.cell
        localEdge = self.localEdge 
        totalEdge = cell[:, localEdge].reshape(-1, localEdge.shape[1])
        return totalEdge

    def total_face(self):
        cell = self.cell
        localFace = self.localFace
        totalFace = cell[:, localFace].reshape(-1, localFace.shape[1])
        return totalFace 
        
    def construct(self):
        NC = self.NC

        totalFace = self.total_face()

        _, i0, j = unique_row(np.sort(totalFace, axis=1))
        self.face = totalFace[i0]

        NF = i0.shape[0]
        self.NF = NF

        self.face2cell = np.zeros((NF, 4), dtype=np.int)

        i1 = np.zeros(NF, dtype=np.int) 
        F = self.F
        i1[j] = np.arange(F*NC)

        self.face2cell[:, 0] = i0//F
        self.face2cell[:, 1] = i1//F
        self.face2cell[:, 2] = i0%F 
        self.face2cell[:, 3] = i1%F 


        totalEdge = self.total_edge()
        self.edge, i2, j = unique_row(np.sort(totalEdge, axis=1))
        E = self.E
        self.cell2edge = np.reshape(j, (NC, E))
        self.NE = self.edge.shape[0]

    def cell_to_node(self):
        """ 
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = np.repeat(range(NC), V)
        val = np.ones(self.V*NC, dtype=np.bool)
        cell2node = csr_matrix((val, (I, cell.flatten())), shape=(NC, NN), dtype=np.bool)
        return cell2node

    def cell_to_edge(self, sparse=False):
        """ The neighbor information of cell to edge
        """
        if sparse == False:
            return self.cell2edge
        else:
            NC = self.NC
            NE = self.NE
            cell2edge = coo_matrix((NC, NE), dtype=np.bool)
            E = self.E
            I = np.repeat(range(NC), E)
            val = np.ones(E*NC, dtype=np.bool)
            cell2edge = csr_matrix((val, (I, self.cell2edge.flatten())), shape=(NC, NE), dtype=np.bool)
            return cell2edge

    def cell_to_edge_sign(self, cell):
        NC = self.NC
        E = self.E
        cell2edgeSign = np.zeros((NC, E), dtype=np.bool)
        localEdge = self.localEdge
        for i, (j, k) in zip(range(E), localEdge):
            cell2edgeSign[:, i] = cell[:, j] < cell[:, k] 
        return cell2edgeSign

    def cell_to_face(self, sparse=False):
        NC = self.NC
        NF = self.NF
        face2cell = self.face2cell
        if sparse == False:
            F = self.F
            cell2face = np.zeros((NC, F), dtype=np.int)
            cell2face[face2cell[:,0], face2cell[:,2]] = range(NF)
            cell2face[face2cell[:,1], face2cell[:,3]] = range(NF)
            return cell2face
        else:
            val = np.ones((2*NF, ), dtype=np.bool)
            I = face2cell[:, [0,1]].flatten()
            J = np.repeat(range(NF), 2)
            cell2face = csr_matrix((val, (I, J)), shape=(NC, NF), dtype=np.bool)
            return cell2face

    def cell_to_cell(self, return_sparse=False, 
            return_boundary=True, return_array=False):
        """ Get the adjacency information of cells
        """
        if return_array:
            return_sparse = False
            return_boundary = False

        NC = self.NC
        NF = self.NF
        face2cell = self.face2cell
        if (return_sparse == False) & (return_array == False):
            F = self.F
            cell2cell = np.zeros((NC, F), dtype=np.int)
            cell2cell[face2cell[:, 0], face2cell[:, 2]] = face2cell[:, 1]
            cell2cell[face2cell[:, 1], face2cell[:, 3]] = face2cell[:, 0]
            return cell2cell
    
        val = np.ones((NF,), dtype=np.bool)
        if return_boundary:
            cell2cell = coo_matrix(
                    (val, (face2cell[:, 0], face2cell[:, 1])),
                    shape=(NC, NC), dtype=np.bool)
            cell2cell += coo_matrix(
                    (val, (face2cell[:, 1], face2cell[:, 0])),
                    shape=(NC, NC), dtype=np.bool)
            return cell2cell.tocsr()
        else:
            isInFace = (face2cell[:, 0] != face2cell[:, 1])
            cell2cell = coo_matrix(
                    (val[isInFace], (face2cell[isInFace, 0], face2cell[isInFace, 1])),
                    shape=(NC, NC), dtype=np.bool)
            cell2cell += coo_matrix(
                    (val[isInFace], (face2cell[isInFace, 1], face2cell[isInFace, 0])),
                    shape=(NC, NC), dtype=np.bool)
            cell2cell = cell2cell.tocsr()
            if return_array == False:
                return cell2cell
            else:
                nn = cell2cell.sum(axis=1).reshape(-1)
                _, adj = cell2cell.nonzero()
                adjLocation = np.zeros(NC+1, dtype=np.int32)
                adjLocation[1:] = np.cumsum(nn)
                return adj.astype(np.int32), adjLocation

    def face_to_node(self, return_sparse=False):

        face = self.face
        FE = self.localFace.shape[1] 
        if return_sparse == False:
            return face
        else:
            N = self.N
            NF = self.NF
            I = np.repeat(range(NF), FE)
            val = np.ones(FE*NF, dtype=np.bool)
            face2node = csr_matrix((val, (I, face)), shape=(NF, N), dtype=np.bool)
            return face2node

    def face_to_edge(self, return_sparse=False):
        cell2edge = self.cell2edge
        face2cell = self.face2cell
        localFace2edge = self.localFace2edge
        FE = localFace2edge.shape[1]
        face2edge = cell2edge[face2cell[:,[0]], localFace2edge[face2cell[:,2]]]
        if return_sparse == False:
            return face2edge
        else:
            NF = self.NF
            NE = self.NE
            I = np.repeat(range(NF), FE)
            J = face2edge.flatten()
            val = np.ones(FE*NF, dtype=np.bool)
            f2e = csr_matrix((val, (I, J)), shape=(NF, NE), dtype=np.bool)
            return f2e

    def face_to_face(self):
        face2edge = self.face_to_edge()
        return face2edge*face2edge.transpose()

    def face_to_cell(self, return_sparse=False):
        if return_sparse==False:
            return self.face2cell
        else:
            NC = self.NC
            NF = self.NF
            I = np.repeat(range(NF), 2)
            J = self.face2cell[:, [0, 1]].flatten()
            val = np.ones(2*NF, dtype=np.bool)
            face2cell = csr_matrix((val, (I, J)), shape=(NF, NC), dtype=np.bool)
            return face2cell 

    def edge_to_node(self, return_sparse=False):
        NN = self.NN
        NE = self.NE
        edge = self.edge
        if return_sparse == False:
            return edge
        else:
            edge = self.edge
            I = np.repeat(range(NE), 2)
            J = edge.flatten()
            val = np.ones(2*NE, dtype=np.bool)
            edge2node = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool)
            return edge2node

    def edge_to_edge(self):
        edge2node = self.edge_to_node()
        return edge2node*edge2node.transpose()

    def edge_to_face(self):
        NF = self.NF
        NE = self.NE
        face2edge = self.face_to_edge()
        FE = face2edge.shape[1]
        I = face2edge.flatten()
        J = np.repeat(range(NF), FE)
        val = np.ones(FE*NF, dtype=np.bool)
        edge2face = csr_matrix((val, (I, J)), shap=(NE, NF), dtype=np.bool)
        return edge2face

    def edge_to_cell(self, localidx=False):
        NC = self.NC
        NE = self.NE
        cell2edge = self.cell2edge
        I = cell2edge.flatten()
        E = self.E
        J = np.repeat(range(NC), E)
        val = np.ones(E*NC, dtype=np.bool)
        edge2cell = csr_matrix((val, (I, J)), shape=(NE, NC), dtype=np.bool)
        return edge2cell

    def node_to_node(self):
        """ The neighbor information of nodes
        """
        NN = self.NN
        NE = self.NE
        edge = self.edge
        I = edge.flatten()
        J = edge[:,[1,0]].flatten()
        val = np.ones((2*NE,), dtype=np.bool)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN),dtype=np.bool)
        return node2node

    def node_to_edge(self):
        NN = self.NN
        NE = self.NE
        
        edge = self.edge
        I = edge.flatten()
        J = np.repeat(range(NE), 2)
        val = np.ones(2*NE, dtype=np.bool)
        node2edge = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool)
        return node2edge

    def node_to_face(self):
        NN = self.NN
        NF = self.NF

        face = self.face
        FV = face.shape[1]

        I = face.flatten()
        J = np.repeat(range(NF), FV)
        val = np.ones(FV*NF, dtype=np.bool)
        node2face = csr_matrix((val, (I, J)), shape=(NF, NN), dtype=np.bool)
        return node2face

    def node_to_cell(self, return_local_index=False):
        """
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = cell.flatten() 
        J = np.repeat(range(NC), V)

        if return_local_index == True:
            val = ranges(V*np.ones(NC, dtype=np.int), start=1) 
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.int)
        else:
            val = np.ones(V*NC, dtype=np.bool)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool)
        return node2cell

    def boundary_node_flag(self):
        NN = self.NN
        face = self.face
        isBdFace = self.boundary_face_flag()
        isBdPoint = np.zeros((NN,), dtype=np.bool)
        isBdPoint[face[isBdFace,:]] = True 
        return isBdPoint 

    def boundary_edge_flag(self):
        NE = self.NE
        face2edge = self.face_to_edge()
        isBdFace = self.boundary_face_flag()
        isBdEdge = np.zeros((NE,), dtype=np.bool)
        isBdEdge[face2edge[isBdFace, :]] = True
        return isBdEdge 

    def boundary_face_flag(self):
        NF = self.NF
        face2cell = self.face_to_cell()
        return face2cell[:, 0] == face2cell[:, 1] 

    def boundary_cell_flag(self):
        NC = self.NC
        face2cell = self.face_to_cell()
        isBdFace = self.boundary_face_flag()
        isBdCell = np.zeros((NC,),dtype=np.bool)
        isBdCell[face2cell[isBdFace, 0]] = True
        return isBdCell 

    def boundary_node_index(self):
        isBdPoint = self.boundary_node_flag()
        idx, = np.nonzero(isBdPoint)
        return idx

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdPoint)
        return idx

    def boundary_face_index(self):
        isBdFace = self.boundary_face_flag()
        idx, = np.nonzero(isBdFace)
        return idx 

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx 
