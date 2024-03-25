from types import ModuleType

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ...common.Tools import ranges
from fealpy.mesh.backup.Mesh import Mesh

class Mesh2d(Mesh):
    def number_of_nodes_of_cells(self):
        return self.ds.number_of_vertices_of_cells()

    def number_of_edges_of_cells(self):
        return self.ds.number_of_edges_of_cells()

    def number_of_faces_of_cells(self):
        return self.ds.number_of_edges_of_cells()

    def number_of_vertices_of_cells(self):
        return self.ds.number_of_vertices_of_cells()

    def top_dimension(self):
        return 2

    def entity(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.ds.cell[index]
        elif etype in {'edge', 'face', 1}:
            return self.ds.edge[index]
        elif etype in {'node', 0}:
            return self.node[index]
        else:
            raise ValueError(f" entity type {etype}  is wrong!")

    def entity_measure(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return 0
        else:
            raise ValueError("`entitytype` is wrong!")

    def entity_barycenter(self, etype=2, index=np.s_[:]):
        node = self.entity('node')
        if etype in {'cell', 2}:
            cell = self.ds.cell
            bc = np.sum(node[cell[index], :], axis=1)/cell.shape[1]
        elif etype in {'edge', 'face', 1}:
            edge = self.ds.edge
            bc = np.sum(node[edge[index], :], axis=1)/edge.shape[1]
        elif etype in {'node', 0}:
            bc = node[index]
        else:
            raise ValueError('the entity `{}` is not correct!'.format(etype))
        return bc

    def node_size(self):
        """
        Notes
        -----
        计算每个网格节点邻接边的长度平均值, 做为节点处的网格尺寸值
        """

        NN = self.number_of_nodes()
        edge = self.entity('edge')
        eh = self.entity_measure('edge')
        h = np.zeros(NN, dtype=self.ftype)
        deg = np.zeros(NN, dtype=self.itype)

        val = np.broadcast_to(eh[:, None], shape=edge.shape)
        np.add.at(h, edge, val)
        np.add.at(deg, edge, 1)

        return h/deg

    def face_unit_normal(self, index=np.s_[:]):
        v = self.face_unit_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def face_unit_tangent(self, index=np.s_[:]):
        edge = self.entity('edge')
        node = self.entity('node')
        NE = self.number_of_edges()
        v = node[edge[index,1],:] - node[edge[index,0],:]
        length = np.sqrt(np.sum(v**2, axis=1))
        v /= length.reshape(-1, 1)
        return v

    def face_normal(self, index=np.s_[:]):
        v = self.face_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def face_tangent(self, index=np.s_[:]):
        node = self.entity('node')
        edge = self.entity('edge')
        v = node[edge[index,1],:] - node[edge[index,0],:]
        return v

    def edge_length(self, index=np.s_[:]):
        node = self.entity('node')
        edge = self.entity('edge')
        v = node[edge[index,1],:] - node[edge[index,0],:]
        length = np.linalg.norm(v, axis=1)
        return length

    def cell_area(self, index=np.s_[:]):
        """
        @brief 根据散度定理计算多边形的面积
        @note 请注意下面的计算方式不方便实现部分单元面积的计算
        """
        NC = self.number_of_cells()
        node = self.entity('node')
        edge = self.entity('edge')
        edge2cell = self.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        v =  node[edge[:, 1], :] - node[edge[:, 0], :]
        val = np.einsum('ij, ij->i', v, node[edge[:, 0], :], optimize=True)
        a = np.bincount(edge2cell[:, 0], weights=val, minlength=NC)
        a+= np.bincount(edge2cell[isInEdge, 1], weights=-val[isInEdge], minlength=NC)
        a /=2
        return a[index]

    def edge_frame(self, index=np.s_[:]):
        t = self.edge_unit_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        n = t@w
        return n, t

    def edge_unit_normal(self, index=np.s_[:]):
        #TODO: 3D Case
        v = self.edge_unit_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def edge_unit_tangent(self, index=np.s_[:]):
        node = self.entity('node')
        edge = self.entity('edge')
        NE = self.number_of_edges()
        v = node[edge[index, -1],:] - node[edge[index, 0],:]
        length = np.linalg.norm(v, axis=1)
        v /= length.reshape(-1, 1)
        return v

    def edge_normal(self, index=np.s_[:]):
        v = self.edge_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def edge_tangent(self, index=np.s_[:]):
        node = self.entity('node')
        edge = self.entity('edge')
        v = node[edge[index, 1],:] - node[edge[index, 0],:]
        return v

    def add_plot(
            self, plot,
            nodecolor='w', edgecolor='k',
            cellcolor=[0.5, 0.9, 0.45], aspect=None,
            linewidths=1, markersize=50,
            showaxis=False, showcolorbar=False,
            cmax=None, cmin=None,
            colorbarshrink=1.0, cmap='jet', box=None):

        from matplotlib.collections import PolyCollection, PatchCollection
        import matplotlib.cm as cm
        from matplotlib import colors
        from matplotlib.patches import Polygon

        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot

        GD = self.geo_dimension()
        if (aspect is None) and (GD == 3):
            axes.set_box_aspect((1, 1, 1))
            axes.set_proj_type('ortho')

        if (aspect is None) and (GD == 2):
            axes.set_box_aspect(1)

        if showaxis == False:
            axes.set_axis_off()
        else:
            axes.set_axis_on()

        if (type(nodecolor) is np.ndarray) & np.isreal(nodecolor[0]):
            cmax = cmax or nodecolor.max()
            cmin = cmin or nodecolor.min()
            norm = colors.Normalize(vmin=cmin, vmax=cmax)
            mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            nodecolor = mapper.to_rgba(nodecolor)

        if isinstance(cellcolor, np.ndarray) & np.isreal(cellcolor[0]):
            cmax = cmax or cellcolor.max()
            cmin = cmin or cellcolor.min()
            norm = colors.Normalize(vmin=cmin, vmax=cmax)
            mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
            mapper.set_array(cellcolor)
            cellcolor = mapper.to_rgba(cellcolor)
            if showcolorbar:
                f = axes.get_figure()
                f.colorbar(mapper, shrink=colorbarshrink, ax=axes)

        node = self.entity('node')
        cell = self.entity('cell')

        if self.meshtype not in {'polygon', 'hepolygon', 'halfedge', 'halfedge2d'}:
            if self.geo_dimension() == 2:
                poly = PolyCollection(node[cell[:, self.ds.ccw], :])
            else:
                import mpl_toolkits.mplot3d as a3
                poly = a3.art3d.Poly3DCollection(node[cell[:, self.ds.ccw], :])
        else:
            if self.meshtype == 'polygon':
                cell, cellLocation = cell
                NC = self.number_of_cells()
                patches = [
                        Polygon(node[cell[cellLocation[i]:cellLocation[i+1]], :], True)
                        for i in range(NC)]
            elif self.ds.NV in {3, 4}:
                NC = self.number_of_cells()
                patches = [
                        Polygon(node[cell[i], :], True)
                        for i in range(NC)]
            else:
                cell, cellLocation = cell
                NC = self.number_of_cells()
                patches = [
                        Polygon(node[cell[cellLocation[i]:cellLocation[i+1]], :], True)
                        for i in range(NC)]
            poly = PatchCollection(patches)

        poly.set_edgecolor(edgecolor)
        poly.set_linewidth(linewidths)
        poly.set_facecolors(cellcolor)

        if box is None:
            if self.geo_dimension() == 2:
                box = np.zeros(4, dtype=np.float64)
            else:
                box = np.zeros(6, dtype=np.float64)

            box[0::2] = np.min(node, axis=0)
            box[1::2] = np.max(node, axis=0)

        tol = np.max(self.entity_measure('edge'))/100
        axes.set_xlim([box[0]-tol, box[1]+0.01]+tol)
        axes.set_ylim([box[2]-tol, box[3]+0.01]+tol)

        if self.geo_dimension() == 3:
            axes.set_zlim(box[4:6])

        return axes.add_collection(poly)


class Mesh2dDataStructure():
    """ The topology data structure of mesh 2d
        This is just a abstract class, and you can not use it directly.
    """

    def __init__(self, NN, cell):
        self.TD = 2
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.itype = cell.dtype
        self.construct()
        self.NF = self.NE

    def reinit(self, NN, cell):
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.construct()

    def clear(self):
        self.edge = None
        self.edge2cell = None

    def number_of_nodes_of_cells(self):
        return self.NVC

    def number_of_edges_of_cells(self):
        return self.NEC

    def number_of_faces_of_cells(self):
        return self.NEC

    def number_of_vertices_of_cells(self):
        return self.NVC

    def total_edge(self):
        NC = self.NC

        cell = self.cell
        localEdge = self.localEdge

        totalEdge = cell[:, localEdge].reshape(-1, 2)
        return totalEdge

    def local_edge(self):
        return self.localEdge

    def construct(self):
        """ Construct edge and edge2cell from cell
        """
        NC = self.NC
        NEC = self.NEC

        totalEdge = self.total_edge()
        _, i0, j = np.unique(np.sort(totalEdge, axis=-1),
                return_index=True,
                return_inverse=True,
                axis=0)
        NE = i0.shape[0]
        self.NE = NE

        self.edge2cell = np.zeros((NE, 4), dtype=self.itype)

        i1 = np.zeros(NE, dtype=self.itype)
        i1[j] = range(NEC*NC)

        self.edge2cell[:, 0] = i0//NEC
        self.edge2cell[:, 1] = i1//NEC
        self.edge2cell[:, 2] = i0%NEC
        self.edge2cell[:, 3] = i1%NEC

        self.edge = totalEdge[i0, :]

    def cell_to_node(self, return_sparse=False):
        """
        """
        NN = self.NN
        NC = self.NC
        NVC = self.NVC
        cell = self.cell

        if return_sparse:
            I = np.repeat(range(NC), NVC)
            val = np.ones(NVC*NC, dtype=np.bool_)
            cell2node = csr_matrix((val, (I, cell.flat)), shape=(NC, NN),
                    dtype=np.bool_)
            return cell2node
        else:
            return cell

    def cell_to_edge(self, return_sparse=False):
        """ The neighbor information of cell to edge
        """
        NE = self.NE
        NC = self.NC
        NEC = self.NEC

        edge2cell = self.edge2cell

        if return_sparse == False:
            cell2edge = np.zeros((NC, NEC), dtype=self.itype)
            cell2edge[edge2cell[:, 0], edge2cell[:, 2]] = np.arange(NE)
            cell2edge[edge2cell[:, 1], edge2cell[:, 3]] = np.arange(NE)
            return cell2edge
        else:
            val = np.ones(2*NE, dtype=np.bool_)
            I = edge2cell[:, [0, 1]].flat
            J = np.repeat(range(NE), 2)
            cell2edge = csr_matrix(
                    (val, (I, J)),
                    shape=(NC, NE), dtype=np.bool_)
            return cell2edge

    def cell_to_edge_sign(self):
        NE = self.NE
        NC = self.NC
        NEC = self.NEC

        edge2cell = self.edge2cell

        cell2edgeSign = np.zeros((NC, NEC), dtype=np.bool_)
        cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = True

        return cell2edgeSign

    def cell_to_face_sign(self):
        return self.cell_to_edge_sign()

    def cell_to_face(self, return_sparse=False):
        """ The neighbor information of cell to edge
        """
        NE = self.NE
        NC = self.NC

        NEC = self.NEC
        edge2cell = self.edge2cell

        if return_sparse == False:
            cell2edge = np.zeros((NC, NEC), dtype=self.itype)
            cell2edge[edge2cell[:, 0], edge2cell[:, 2]] = np.arange(NE)
            cell2edge[edge2cell[:, 1], edge2cell[:, 3]] = np.arange(NE)
            return cell2edge
        else:
            val = np.ones(2*NE, dtype=np.bool_)
            I = edge2cell[:, [0, 1]].flat
            J = np.repeat(range(NE), 2)
            cell2edge = csr_matrix(
                    (val, (I, J)),
                    shape=(NC, NE))
            return cell2edge


    def cell_to_cell(self, return_sparse=False, return_boundary=True, return_array=False):
        """ Consctruct the neighbor information of cells
        """
        if return_array:
             return_sparse = False
             return_boundary = False

        NC = self.NC
        edge2cell = self.edge2cell
        if (return_sparse == False) & (return_array == False):
            NEC = self.NEC
            cell2cell = np.zeros((NC, NEC), dtype=self.itype)
            cell2cell[edge2cell[:, 0], edge2cell[:, 2]] = edge2cell[:, 1]
            cell2cell[edge2cell[:, 1], edge2cell[:, 3]] = edge2cell[:, 0]
            return cell2cell
        NE = self.NE
        val = np.ones((NE,), dtype=np.bool_)
        if return_boundary:
            cell2cell = coo_matrix(
                    (val, (edge2cell[:, 0], edge2cell[:, 1])),
                    shape=(NC, NC), dtype=np.bool_)
            cell2cell += coo_matrix(
                    (val, (edge2cell[:, 1], edge2cell[:, 0])),
                    shape=(NC, NC), dtype=np.bool_)
            return cell2cell.tocsr()
        else:
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            cell2cell = coo_matrix(
                    (val[isInEdge], (edge2cell[isInEdge, 0], edge2cell[isInEdge, 1])),
                    shape=(NC, NC), dtype=np.bool_)
            cell2cell += coo_matrix(
                    (val[isInEdge], (edge2cell[isInEdge, 1], edge2cell[isInEdge, 0])),
                    shape=(NC, NC), dtype=np.bool_)
            cell2cell = cell2cell.tocsr()
            if return_array == False:
                return cell2cell
            else:
                nn = cell2cell.sum(axis=1).reshape(-1)
                _, adj = cell2cell.nonzero()
                adjLocation = np.zeros(NC+1, dtype=np.int32)
                adjLocation[1:] = np.cumsum(nn)
                return adj.astype(np.int32), adjLocation

    def edge_to_node(self, return_sparse=False):
        NN = self.NN
        NE = self.NE

        edge = self.edge
        if return_sparse == False:
            return edge
        else:
            NVE = self.NVE
            I = np.repeat(range(NE), NVE)
            J = edge.flat
            val = np.ones(NVE*NE, dtype=np.bool_)
            edge2node = csr_matrix((val, (I, J)), shape=(NE, NN))
            return edge2node

    def edge_to_edge(self):
        edge2node = self.edge_to_node(return_sparse=True)
        return edge2node*edge2node.T

    def edge_to_cell(self, return_sparse=False):
        if return_sparse==False:
            return self.edge2cell
        else:
            NC = self.NC
            NE = self.NE
            I = np.repeat(range(NE), 2)
            J = self.edge2cell[:, [0, 1]].flat
            val = np.ones(2*NE, dtype=np.bool_)
            face2cell = csr_matrix((val, (I, J)), shape=(NE, NC))
            return face2cell

    def face_to_cell(self, return_sparse=False):
        return self.edge_to_cell(return_sparse=return_sparse)

    def node_to_node(self, return_array=False):

        """
        Notes
        -----
            节点与节点的相邻关系

        TODO
        ----
            曲边元的边包含两个以上的点,
        """

        NN = self.NN
        NE = self.NE
        edge = self.edge
        NVE = self.NVE
        I = edge[:, [0, -1]].flat
        J = edge[:, [-1, 0]].flat
        val = np.ones((2*NE,), dtype=np.bool_)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN))
        if return_array == False:
            return node2node
        else:
            nn = node2node.sum(axis=1).reshape(-1)
            _, adj = node2node.nonzero()
            adjLocation = np.zeros(NN+1, dtype=np.int32)
            adjLocation[1:] = np.cumsum(nn)
            return adj.astype(np.int32), adjLocation

    def node_to_node_in_edge(self, NN, edge):
        """
        Notes
        ----
        TODO
        """
        I = edge.flatten()
        J = edge[:, [1, 0]].flatten()
        val = np.ones(2*edge.shape[0], dtype=np.bool_)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN), dtype=np.bool_)
        return node2node

    def node_to_edge(self):
        """
        """
        NN = self.NN
        NE = self.NE
        NVE = self.NVE
        I = self.edge.flat
        J = np.repeat(range(NE), NVE)
        val = np.ones(NVE*NE, dtype=np.bool_)
        node2edge = csr_matrix((val, (I, J)), shape=(NN, NE))
        return node2edge

    def node_to_cell(self, return_localidx=False):
        """
        """
        NN = self.NN
        NC = self.NC
        NVC = self.NVC

        I = self.cell.flat
        J = np.repeat(range(NC), NVC)

        if return_localidx == False:
            val = np.ones(NVC*NC, dtype=np.bool_)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC))
        else:
            val = ranges(NVC*np.ones(NC, dtype=self.itype), start=1)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=self.itype)
        return node2cell

    def boundary_edge_to_edge(self):
        """
        """
        NN = self.NN
        edge = self.edge
        index = self.boundary_edge_index()
        bdEdge = edge[index]
        n = bdEdge.shape[0]
        val = np.ones(n, dtype=np.bool_)
        m0 = csr_matrix((val, (range(n), bdEdge[:, 0])), shape=(n, NN))
        m1 = csr_matrix((val, (range(n), bdEdge[:, 1])), shape=(n, NN))
        _, pre = (m0*m1.T).nonzero()
        _, nex = (m1*m0.T).nonzero()
        return index[pre], index[nex]

    def boundary_node_flag(self):
        NN = self.NN
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdNode = np.zeros((NN,), dtype=np.bool_)
        isBdNode[edge[isBdEdge,:]] = True
        return isBdNode

    def boundary_edge_flag(self):
        edge2cell = self.edge2cell
        return edge2cell[:, 0] == edge2cell[:, 1]

    def boundary_edge(self):
        edge = self.edge
        return edge[self.boundary_edge_index()]

    def boundary_face_flag(self):
        edge2cell = self.edge2cell
        return edge2cell[:, 0] == edge2cell[:, 1]

    def boundary_face(self):
        edge = self.edge
        return edge[self.boundary_edge_index()]

    def boundary_cell_flag(self):
        NC = self.NC
        edge2cell = self.edge2cell
        isBdCell = np.zeros((NC,), dtype=np.bool_)
        isBdEdge = self.boundary_edge_flag()
        isBdCell[edge2cell[isBdEdge,0]] = True
        return isBdCell

    def boundary_node_index(self):
        isBdPoint = self.boundary_node_flag()
        idx, = np.nonzero(isBdPoint)
        return idx

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx

    def boundary_face_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx

