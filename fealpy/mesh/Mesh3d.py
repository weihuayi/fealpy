import numpy as np

from types import ModuleType
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from ..common import ranges
from .Mesh import Mesh


class Mesh3d(Mesh):

    def number_of_nodes_of_cells(self):
        return self.ds.number_of_nodes_of_cells()

    def number_of_edges_of_cells(self):
        return self.ds.number_of_edges_of_cells()

    def number_of_faces_of_cells(self):
        return self.ds.number_of_faces_of_cells()

    def geo_dimension(self):
        return 3 

    def top_dimension(self):
        """
        @brief
        """
        return 3

    def entity(self, etype='cell', index=np.s_[:]):
        if etype in {'cell', 3}:
            return self.ds.cell[index]
        elif etype in {'face', 2}:
            return self.ds.face[index]
        elif etype in {'edge', 1}:
            return self.ds.edge[index]
        elif etype in {'node', 0}:
            return self.node[index]
        else:
            raise ValueError("`etype` is wrong!")

    def entity_measure(self, etype=3, index=np.s_[:]):
        if etype in {'cell', 3}:
            return self.cell_volume(index=index)
        elif etype in {'face', 2}:
            return self.face_area(index=index)
        elif etype in {'edge', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return np.zeros(1, dtype=self.ftype)
        else:
            raise ValueError("`entitytype` is wrong!")

    def entity_barycenter(self, etype='cell', index=np.s_[:]):
        node = self.node
        if etype in {'cell', 3}:
            cell = self.ds.cell
            bc = np.sum(node[cell[index], :], axis=1).reshape(-1, 3)/cell.shape[1]
        elif etype in {'face', 2}:
            face = self.ds.face
            bc = np.sum(node[face[index], :], axis=1).reshape(-1, 3)/face.shape[1]
        elif etype in {'edge', 1}:
            edge = self.ds.edge
            bc = np.sum(node[edge[index], :], axis=1).reshape(-1, 3)/edge.shape[1]
        elif etype in {'node', 0}:
            bc = node[index]
        else:
            raise ValueError("`etype` is wrong!")

        return bc

    def edge_tangent(self):
        edge = self.ds.edge
        node = self.node
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        return v

    def edge_unit_tangent(self):
        edge = self.ds.edge
        node = self.node
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        length = np.sqrt(np.square(v).sum(axis=1))
        return v/length.reshape(-1, 1)

    def add_plot(
            self, plot,
            nodecolor='k', edgecolor='k', facecolor='w', cellcolor='w',
            aspect=[1, 1, 1],
            linewidths=0.5, markersize=20,
            showaxis=False, alpha=0.8, shownode=False, showedge=False, threshold=None):

        if isinstance(plot, ModuleType):
            from mpl_toolkits.mplot3d import Axes3D
            fig = plot.figure()
            axes = fig.add_subplot(111, projection='3d')
        else:
            axes = plot
        axes.set_box_aspect(aspect)
        axes.set_proj_type('ortho')

        if showaxis == False:
            axes.set_axis_off()
        else:
            axes.set_axis_on()

        if (type(nodecolor) is np.ndarray) & np.isreal(nodecolor[0]):
            cmax = nodecolor.max()
            cmin = nodecolor.min()
            norm = colors.Normalize(vmin=cmin, vmax=cmax)
            mapper = cm.ScalarMappable(norm=norm, cmap='rainbow')
            nodecolor = mapper.to_rgba(nodecolor)

        node = self.node
        if shownode:
            axes.scatter(
                    node[:, 0], node[:, 1], node[:, 2],
                    color=nodecolor, s=markersize)

        if showedge:
            edge = self.ds.edge
            vts = node[edge]
            edges = a3.art3d.Line3DCollection(
                   vts,
                   linewidths=linewidths,
                   color=edgecolor)
            return axes.add_collection3d(edges)

        face = self.entity('face')
        isBdFace = self.ds.boundary_face_flag()
        if threshold is None:
            face = face[isBdFace][:, self.ds.ccw]
        else:
            bc = self.entity_barycenter('cell')
            isKeepCell = threshold(bc)
            face2cell = self.ds.face_to_cell()
            isInterfaceFace = np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 1
            isBdFace = (np.sum(isKeepCell[face2cell[:, 0:2]], axis=-1) == 2) & isBdFace
            face = face[isBdFace | isInterfaceFace][:, self.ds.ccw]
        
        import mpl_toolkits.mplot3d as a3
        faces = a3.art3d.Poly3DCollection(
                node[face],
                facecolor=facecolor,
                linewidths=linewidths,
                edgecolor=edgecolor,
                alpha=alpha)
        h = axes.add_collection3d(faces)
        box = np.zeros((2, 3), dtype=np.float64)
        box[0, :] = np.min(node, axis=0)
        box[1, :] = np.max(node, axis=0)
        axes.scatter(box[:, 0], box[:, 1], box[:, 2], s=0)
        return h


class Mesh3dDataStructure():
    def __init__(self, NN, cell):
        self.itype = cell.dtype
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
        self.edge = None
        self.face2cell = None
        self.cell2edge = None

    def number_of_nodes_of_cells(self):
        return self.NVC

    def number_of_edges_of_cells(self):
        return self.NEC

    def number_of_faces_of_cells(self):
        return self.NFC

    def total_edge(self):
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

        _, i0, j = np.unique(
                np.sort(totalFace, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0)

        self.face = totalFace[i0]

        NF = i0.shape[0]
        self.NF = NF

        self.face2cell = np.zeros((NF, 4), dtype=self.itype)

        i1 = np.zeros(NF, dtype=self.itype)
        NFC = self.NFC
        i1[j] = np.arange(NFC*NC)

        self.face2cell[:, 0] = i0 // NFC
        self.face2cell[:, 1] = i1 // NFC
        self.face2cell[:, 2] = i0 % NFC
        self.face2cell[:, 3] = i1 % NFC

        totalEdge = self.total_edge()
        self.edge, i2, j = np.unique(
                np.sort(totalEdge, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0)
        NEC = self.NEC
        self.cell2edge = np.reshape(j, (NC, NEC))
        self.NE = self.edge.shape[0]

    def cell_to_node(self, return_sparse=False):
        """
        """
        NN = self.NN
        NC = self.NC
        NVC = self.NVC

        cell = self.cell

        if return_sparse is False:
            return cell
        else:
            cell2node = csr_matrix(
                    (
                        np.ones(NVC*NC, dtype=np.bool_),
                        (
                            np.repeat(range(NC), NVC),
                            cell.flat
                        )
                    ), shape=(NC, NN))
        return cell2node

    def cell_to_edge(self, return_sparse=False):
        """ The neighbor information of cell to edge
        """
        if return_sparse is False:
            return self.cell2edge
        else:
            NC = self.NC
            NE = self.NE
            cell2edge = coo_matrix((NC, NE), dtype=np.bool_)
            NEC = self.NEC
            cell2edge = csr_matrix(
                    (
                        np.ones(NEC*NC, dtype=np.bool_),
                        (
                            np.repeat(range(NC), NEC),
                            self.cell2edge.flat
                        )
                    ), shape=(NC, NE))
            return cell2edge

    def cell_to_edge_sign(self, cell=None):
        """

        TODO: check here
        """
        if cell==None:
            cell = self.cell
        NC = self.NC
        NEC = self.NEC
        cell2edgeSign = np.zeros((NC, NEC), dtype=np.bool_)
        localEdge = self.localEdge
        E = localEdge.shape[0]
        for i, (j, k) in zip(range(E), localEdge):
            cell2edgeSign[:, i] = cell[:, j] < cell[:, k]
        return cell2edgeSign

    def cell_to_face_sign(self):
        """
        """
        NC = self.NC
        NF = self.NF
        face2cell = self.face2cell
        NFC = self.NFC
        cell2facesign = np.zeros((NC, NFC), dtype=self.bool_)
        cell2facesign[face2cell[:, 0], face2cell[:, 2]] = True 
        return cell2facesign

    def cell_to_face(self, return_sparse=False):
        NC = self.NC
        NF = self.NF
        face2cell = self.face2cell
        if return_sparse is False:
            NFC = self.NFC
            cell2face = np.zeros((NC, NFC), dtype=self.itype)
            cell2face[face2cell[:, 0], face2cell[:, 2]] = range(NF)
            cell2face[face2cell[:, 1], face2cell[:, 3]] = range(NF)
            return cell2face
        else:
            cell2face = csr_matrix(
                    (
                        np.ones((2*NF, ), dtype=np.bool_),
                        (
                            face2cell[:, [0, 1]].flat,
                            np.repeat(range(NF), 2)
                        )
                    ), shape=(NC, NF))
            return cell2face

    def cell_to_cell(
            self, return_sparse=False,
            return_boundary=True, return_array=False):
        """ Get the adjacency information of cells
        """
        if return_array:
            return_sparse = False
            return_boundary = False

        NC = self.NC
        NF = self.NF
        face2cell = self.face2cell
        if (return_sparse is False) & (return_array is False):
            NFC = self.NFC
            cell2cell = np.zeros((NC, NFC), dtype=self.itype)
            cell2cell[face2cell[:, 0], face2cell[:, 2]] = face2cell[:, 1]
            cell2cell[face2cell[:, 1], face2cell[:, 3]] = face2cell[:, 0]
            return cell2cell

        val = np.ones((NF,), dtype=np.bool_)
        if return_boundary:
            cell2cell = coo_matrix(
                    (val, (face2cell[:, 0], face2cell[:, 1])),
                    shape=(NC, NC))
            cell2cell += coo_matrix(
                    (val, (face2cell[:, 1], face2cell[:, 0])),
                    shape=(NC, NC))
            return cell2cell.tocsr()
        else:
            isInFace = (face2cell[:, 0] != face2cell[:, 1])
            cell2cell = coo_matrix(
                    (
                        val[isInFace],
                        (face2cell[isInFace, 0], face2cell[isInFace, 1])
                    ),
                    shape=(NC, NC), dtype=np.bool_)
            cell2cell += coo_matrix(
                    (
                        val[isInFace],
                        (face2cell[isInFace, 1], face2cell[isInFace, 0])
                    ), shape=(NC, NC), dtype=np.bool_)
            cell2cell = cell2cell.tocsr()
            if return_array is False:
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
        if return_sparse is False:
            return face
        else:
            NN = self.NN
            NF = self.NF
            face2node = csr_matrix(
                    (
                        np.ones(FE*NF, dtype=np.bool_),
                        (
                            np.repeat(range(NF), FE),
                            face.flat
                        )
                    ), shape=(NF, NN), dtype=np.bool_)
            return face2node

    def face_to_edge(self, return_sparse=False):
        cell2edge = self.cell2edge
        face2cell = self.face2cell
        localFace2edge = self.localFace2edge
        face2edge = cell2edge[
                face2cell[:, [0]],
                localFace2edge[face2cell[:, 2]]
                ]
        if return_sparse is False:
            return face2edge
        else:
            NF = self.NF
            NE = self.NE
            NEF = self.NEF
            f2e = csr_matrix(
                    (
                        np.ones(NEF*NF, dtype=np.bool_),
                        (
                            np.repeat(range(NF), NEF),
                            face2edge.flat
                        )
                    ), shape=(NF, NE))
            return f2e

    def face_to_face(self):
        face2edge = self.face_to_edge(return_sparse=True)
        return face2edge*face2edge.T

    def face_to_cell(self, return_sparse=False):
        if return_sparse is False:
            return self.face2cell
        else:
            NC = self.NC
            NF = self.NF
            face2cell = csr_matrix(
                    (
                        np.ones(2*NF, dtype=np.bool_),
                        (
                            np.repeat(range(NF), 2),
                            self.face2cell[:, [0, 1]].flat
                        )
                    ), shape=(NF, NC), dtype=np.bool_)
            return face2cell

    def edge_to_node(self, return_sparse=False):
        NN = self.NN
        NE = self.NE
        edge = self.edge
        if return_sparse is False:
            return edge
        else:
            edge = self.edge
            edge2node = csr_matrix(
                    (
                        np.ones(2*NE, dtype=np.bool_),
                        (
                            np.repeat(range(NE), 2),
                            edge.flat
                        )
                    ), shape=(NE, NN), dtype=np.bool_)
            return edge2node

    def edge_to_edge(self):
        edge2node = self.edge_to_node(return_sparse=True)
        return edge2node*edge2node.T

    def edge_to_face(self):
        NF = self.NF
        NE = self.NE
        face2edge = self.face_to_edge()
        NEF = self.NEF
        edge2face = csr_matrix(
                (
                    np.ones(NEF*NF, dtype=np.bool_),
                    (
                        face2edge.flat,
                        np.repeat(range(NF), NEF)
                    )
                ), shape=(NE, NF))
        return edge2face

    def edge_to_cell(self, return_localidx=False):
        NC = self.NC
        NE = self.NE
        cell2edge = self.cell2edge
        NEC = self.NEC

        if return_localidx is False:
            edge2cell = csr_matrix(
                    (
                        np.ones(NEC*NC, dtype=np.bool_),
                        (
                            cell2edge.flat,
                            np.repeat(range(NC), NEC)
                        )
                    ), shape=(NE, NC))
        else:
            raise ValueError("Need to implement!")

        return edge2cell

    def node_to_node(self):
        """ The neighbor information of nodes
        """
        NN = self.NN
        NE = self.NE
        edge = self.edge
        node2node = csr_matrix(
                (
                    np.ones((2*NE,), dtype=np.bool_),
                    (
                        edge.flat,
                        edge[:, [1, 0]].flat
                    )
                ), shape=(NN, NN), dtype=np.bool_)
        return node2node

    def node_to_edge(self):
        NN = self.NN
        NE = self.NE
        edge = self.edge
        node2edge = csr_matrix(
                (
                    np.ones(2*NE, dtype=np.bool_),
                    (
                        edge.flat,
                        np.repeat(range(NE), 2)
                    )
                ), shape=(NN, NE))
        return node2edge

    def node_to_face(self):
        NN = self.NN
        NF = self.NF

        face = self.face
        NVF = self.NVF
        node2face = csr_matrix(
                (
                    np.ones(NVF*NF, dtype=np.bool_),
                    (
                        face.flat,
                        np.repeat(range(NF), NVF)
                    )
                ), shape=(NN, NF))
        return node2face

    def node_to_cell(self, return_localidx=False):
        """
        """
        NN = self.NN
        NC = self.NC
        NVC = self.NVC

        cell = self.cell

        if return_localidx is False:
            node2cell = csr_matrix(
                    (
                        np.ones(NVC*NC, dtype=np.bool_),
                        (
                            cell.flatten(),
                            np.repeat(range(NC), NVC)
                        )
                    ), shape=(NN, NC), dtype=np.bool_)
        else:
            node2cell = csr_matrix(
                    (
                        ranges(NVC*np.ones(NC, dtype=self.itype), start=1),
                        (
                            cell.flatten(),
                            np.repeat(range(NC), NVC)
                        )
                    ), shape=(NN, NC), dtype=self.itype)
        return node2cell

    def boundary_node_flag(self):
        NN = self.NN
        face = self.face
        isBdFace = self.boundary_face_flag()
        isBdPoint = np.zeros((NN,), dtype=np.bool_)
        isBdPoint[face[isBdFace, :]] = True
        return isBdPoint

    def boundary_edge_flag(self):
        NE = self.NE
        face2edge = self.face_to_edge()
        isBdFace = self.boundary_face_flag()
        isBdEdge = np.zeros((NE,), dtype=np.bool_)
        isBdEdge[face2edge[isBdFace, :]] = True
        return isBdEdge

    def boundary_face_flag(self):
        face2cell = self.face_to_cell()
        return face2cell[:, 0] == face2cell[:, 1]

    def boundary_cell_flag(self):
        NC = self.NC
        face2cell = self.face_to_cell()
        isBdFace = self.boundary_face_flag()
        isBdCell = np.zeros((NC,), dtype=np.bool_)
        isBdCell[face2cell[isBdFace, 0]] = True
        return isBdCell

    def boundary_node_index(self):
        isBdNode = self.boundary_node_flag()
        idx, = np.nonzero(isBdNode)
        return idx

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx

    def boundary_face_index(self):
        isBdFace = self.boundary_face_flag()
        idx, = np.nonzero(isBdFace)
        return idx

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx
