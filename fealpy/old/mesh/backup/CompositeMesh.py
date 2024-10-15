import numpy as np
from .core import multi_index_matrix
from .core import lagrange_shape_function

"""
线性组合网格

inter
tri
quad
tet
hex
prism
"""

class CompositeMesh:
    def __init__(self, node, cells):

        NN = node.shape[0]
        GD = node.shape[1]

        if GD < 3:
            self.node = np.zeros((NN, 3), dtype=node.dtype)
            self.node[:, 0:GD] = node
        else:
            self.node = node

    def geo_dimension(self):
        return self.GD


    def add_plot(self, axes):

class CompositeMeshDataStructure:
    def __init__(self, NN, cells):
        self.NN = NN
        self.cells = cells

if __name__ == "__main__":
    from fealpy.mesh import MeshFactory as MF
    box = [
