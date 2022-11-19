import numpy as np

from fealpy.quadrature import GaussLegendreQuadrature

class TrussMesh():
    def __init__(self, node, edge):
        self.node = node
        self.itype = edge.dtype
        self.ftype = node.dtype

        self.meshtype = 'truss'
        
        self.NN = node.shape[0]
        self.ds = TrussMeshDataStructure(self.NN, edge)

        self.nodedata = {}
        self.edgedata = {}

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

    def number_of_edges(self):
        return self.ds.edge.shape[0]

    def entity(self, etype='node'):
        if etype in {'node', 0}:
            return self.node
        elif etype in {'edge', 1}:
            return self.ds.edge
    
    def lagrange_dof(self, p, spacetype='C'):
        return CLagrangeTrussDof2d(self, p)


    def entity_measure(self, etype='node'):
        if etype in {'node', 0}:
            return 0.0 
        elif etype in {'edge', 1}:
            return self.edge_length() 

    def bc_to_point(self, bc, index=np.s_[:], node=None):
        """

        Notes
        -----
            把重心坐标转换为实际空间坐标
        """
        node = self.node if node is None else node
        edge = self.entity('edge')
        p = np.einsum('...j, ijk->...ik', bc, node[edge[index]])
        return p

    def edge_length(self):
        node = self.entity('node')
        edge = self.entity('edge')

        v = node[edge[:, 0]] - node[edge[:, 1]]
        h = np.sqrt(np.sum(v**2, axis=1))
        return h

    def unit_edge_tangent(self):
        node = self.entity('node')
        edge = self.entity('edge')
        v = node[edge[:, 0]] - node[edge[:, 1]]
        h = np.sqrt(np.sum(v**2, axis=1))
        return v/h[:, None]
    
    def add_plot(self, axes, 
            nodecolor='r',
            edgecolor='k', 
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

        edge = self.entity('edge') 

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

        vts = node[edge]
        edges = a3.art3d.Line3DCollection(
               vts,
               linewidths=linewidths,
               color=edgecolor)
        return axes.add_collection3d(edges)


class TrussMeshDataStructure():

    def __init__(self, NN, edge):
        self.NN = NN
        self.edge = edge
        self.NE = len(edge)

    def edge_to_node(self):
        return self.edge

    def edge_to_edge(self):
        NE = self.NE
        node2edge = self.node2edge
        edge2edge = np.zeros((NE, 2), dtype=np.int)
        edge2edge[node2edge[:, 0], node2edge[:, 2]] = node2edge[:, 1]
        edge2edge[node2edge[:, 1], node2edge[:, 3]] = node2edge[:, 0]
        return edge2edge

    def node_to_edge(self):
        NN = self.NN
        NE = self.NE
        I = self.edge.flat
        J = np.repeat(range(NE), 2)
        val = np.ones(2*NE, dtype=np.bool_)
        node2edge = csr_matrix((val, (I, J)), shape=(NN, NE))
        return node2edge
