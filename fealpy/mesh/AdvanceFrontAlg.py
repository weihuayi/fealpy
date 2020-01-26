
import numpy as np

class Domain():
    def __init__(self, node, edge):
        self.node = node
        self.edge = edge


    def number_of_nodes(self):
        return self.node.shape[0]

    def number_of_edges(self):
        return self.edge.shape[0]

    def normal(self):
        v = self.tangent(index=index)
        w = np.array([(0, -1),(1, 0)])
        return v@w

    def tangent(self, index=None):
        node = self.node
        edge = self.edge
        index = index if index is not None else np.s_[:]
        v = node[edge[index, 1]] - node[edge[index, 0]]
        return v

    def uniform_refine(self, n=1):
        for i in range(n):
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            node = self.node
            edge = self.edge
            edge2newNode = np.arange(NN, NN+NE)
            newNode = (node[edge[:, 0]] + node[edge[:, 1]])/2
            self.node = np.r_['0', node, newNode]
            self.edge = np.zeros((2*NE, 2), dtype=np.int)
            self.cell[0:NE, 0] = cell[:, 0]
            self.cell[0:NE, 1] = range(NN, NN+NE)
            self.cell[NE:, 0] = range(NN, NN+NE)
            self.cell[NE:, 1] = cell[:, 1]


class AdvancingFrontAlg():
    def __init__(self, domain):
        self.domain = domain

    def update_front(self):
        pass

    def smooth_front(self):
        pass

    def insert_points(self):
        pass

    def remove_bad_points(self):
        pass
        

