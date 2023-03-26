import numpy as np

from fealpy.quadrature import GaussLegendreQuadrature

class EdgeMesh():
    def __init__(self, node, cell):
        self.node = node
        self.itype = cell.dtype
        self.ftype = node.dtype

        self.meshtype = 'edge'
        
        self.NN = node.shape[0]
        self.ds = EdgeMeshDataStructure(self.NN, cell)

        self.nodedata = {}
        self.celldata = {}
        self.edgedata = self.celldata
        self.facedata = self.celldata

    def geo_dimension(self):
        return self.node.shape[1]

    def integrator(self, k, etype='cell'):
        """

        Notes
        -----
            返回第 k 个高斯积分公式。
        """
        return GaussLegendreQuadrature(k)

    def number_of_nodes(self):
        return self.ds.NN

    def number_of_cell(self):
        return self.ds.cell.shape[0]

    def entity(self, etype='node'):
        if etype in {'node', 0}:
            return self.node
        elif etype in {'edge', 'face', 'cell', 1}:
            return self.ds.cell
    
    def lagrange_dof(self, p, spacetype='C'):
        return CLagrangeTrussDof2d(self, p)


    def entity_measure(self, etype='node'):
        if etype in {'node', 0}:
            return 0.0 
        elif etype in {'edge', 'face', 'cell', 1}:
            return self.cell_length() 

    def bc_to_point(self, bc, index=np.s_[:], node=None):
        """

        Notes
        -----
            把重心坐标转换为实际空间坐标
        """
        node = self.node if node is None else node
        cell = self.entity('cell')
        p = np.einsum('...j, ijk->...ik', bc, node[cell[index]])
        return p

    def cell_length(self):
        node = self.entity('node')
        cell = self.entity('cell')

        v = node[cell[:, 0]] - node[cell[:, 1]]
        h = np.sqrt(np.sum(v**2, axis=1))
        return h

    def cell_unit_tangent(self):
        node = self.entity('node')
        cell = self.entity('cell')
        v = node[cell[:, 0]] - node[cell[:, 1]]
        h = np.sqrt(np.sum(v**2, axis=1))
        return v/h[:, None]
    
    def add_plot(self, axes, 
            nodecolor='r',
            cellcolor='k', 
            linewidths=1, 
            aspect='equal',
            markersize=10,
            box=None,
            disp=None,
            scale=1.0
            ):

        import mpl_toolkits.mplot3d as a3
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        try:
            axes.set_aspect(aspect)
        except NotImplementedError:
            pass

        GD = self.geo_dimension()
        node = self.entity('node')
        if GD == 2:
            axes.scatter(node[:, 0], node[:, 1], c=nodecolor, s=markersize)
        else:
            axes.scatter(node[:, 0], node[:, 1], node[:, 2], c=nodecolor, s=markersize)

        cell = self.entity('cell') 

        if box is None:
            if self.geo_dimension() == 2:
                box = np.zeros(4, dtype=np.float64)
            else:
                box = np.zeros(6, dtype=np.float64)

        box[0::2] = np.min(node, axis=0)
        box[1::2] = np.max(node, axis=0)

        axes.set_xlim([box[0], box[1]+0.01])
        axes.set_ylim([box[2]-0.01, box[3]])

        if GD == 3:
            axes.set_zlim(box[4:6])

        vts = node[cell]
        cells = a3.art3d.Line3DCollection(
               vts,
               linewidths=linewidths,
               color=cellcolor)
        return axes.add_collection3d(cells)


class EdgeMeshDataStructure():

    def __init__(self, NN, cell):
        self.NN = NN
        self.cell = cell
        self.NC = len(cell)
        self.NE = self.NC
        self.NF = self.NC

    def cell_to_node(self):
        return self.cell

    def cell_to_cell(self):
        NC = self.NC
        node2cell = self.node2cell
        cell2cell = np.zeros((NC, 2), dtype=np.int)
        cell2cell[node2cell[:, 0], node2cell[:, 2]] = node2cell[:, 1]
        cell2cell[node2cell[:, 1], node2cell[:, 3]] = node2cell[:, 0]
        return cell2cell

    def node_to_cell(self):
        NN = self.NN
        NC = self.NC
        I = self.cell.flat
        J = np.repeat(range(NC), 2)
        val = np.ones(2*NC, dtype=np.bool_)
        node2cell = csr_matrix((val, (I, J)), shape=(NN, NC))
        return node2cell
