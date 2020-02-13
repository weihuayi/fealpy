import numpy as np
from .mesh_tools import unique_row, find_node, find_entity, show_mesh_1d
from types import ModuleType
from ..quadrature import GaussLegendreQuadrature

class HalfEdgeDomain():
    def __init__(self, node, halfedge, NC):
        self.node = node
        self.halfedge = halfedge

        self.NN = node.shape[0]
        self.NE = halfedge.shape[0]//2
        self.NC = NC # number of subdomain

        self.ftype  = node.dtype
        self.itype = halfedge.dtype

        self.nodedata = {}

    def entity(self, etype=1):
        if etype == 'halfedge':
            return self.halfedge
        elif etype == 'edge':
            halfedge = self.halfedge
            isMainHEdge = halfedge[:, 5] == 1
            edge = np.zeros((NE, 2), dtype=self.itype)
            edge[:, 0] = halfedge[halfedge[isMainHEdge, 4], 0]
            edge[:, 1] = halfedge[isMainHEdge, 0]
            return edge
        elif etype in 'node':
            return self.node
        else:
            raise ValueError("`entitytype` is wrong!")

    def edge_length(self):
        node = self.entity('node') 
        edge = self.entity('edge')
        v = node[edge[:, 1]] - node[edge[:, 0]]
        return np.sqrt(np.sum(v**2, axis=-1))

    def edge_normal(self, index=None):
        v = self.edge_tangent(index=index)
        w = np.array([(0, -1),(1, 0)])
        return v@w

    def edge_tangent(self):
        node = self.node
        halfedge = self.halfedge
        e1 = halfedge[:, 0]
        e0 = halfedge[halfedge[:, 4], 0]
        v = node[e1] - node[e0]
        return v

    def geo_dimension(self):
        return 2 

    def top_dimension(self):
        return 1

    def uniform_refine(self):
        pass

    def number_of_nodes(self):
        return self.node.shape[0]

    def number_of_edges(self):
        return self.halfedge.shape[0]

    def add_plot(self, plot,
            nodecolor='k', edgecolor='k',
            aspect='equal', linewidths=0.1, markersize=10,
            showaxis=False):
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot
        return show_mesh_1d(axes, self,
                nodecolor=nodecolor, cellcolor=edgecolor, aspect=aspect,
                linewidths=linewidths, markersize=markersize,
                showaxis=showaxis)

    def find_node(self, axes, node=None,
            index=None, showindex=False,
            color='r', markersize=20,
            fontsize=10, fontcolor='r'):

        if node is None:
            node = self.node
        find_node(axes, node,
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    def find_edge(self, axes,
            index=None, showindex=False,
            color='m', markersize=25,
            fontsize=15, fontcolor='k'):
        find_entity(axes, self, entity='cell',
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)
