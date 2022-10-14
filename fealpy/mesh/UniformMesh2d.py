
import numpy as np
from .Mesh2d import Mesh2d
from .StructureMesh2dDataStructure import StructureMesh2dDataStructure

"""

"""

class UniformMesh2d(Mesh2d):
    def __init__(self, extent, h=(1.0, 1.0), origin=(0.0, 0.0)):
        self.extent = extent
        self.h = h 
        self.origin = origin

        nx = extent[1] - extent[0]
        ny = extent[3] - extent[2]
        self.ds = StructureMesh2dDataStructure(nx, ny)

    @property
    def node(self):
        NN = self.ds.NN
        nx = self.ds.nx
        ny = self.ds.ny
        box = [origin[0], origin[0] + nx*self.h[0], origin[1],   origin[1] + ny*self.h[1]]

        X, Y = np.mgrid[
                box[0]:box[1]:complex(0, nx+1),
                box[2]:box[3]:complex(0, ny+1)]
        node = np.zeros((NN, 2), dtype=self.ftype)
        node[:, 0] = X.flat
        node[:, 1] = Y.flat
        return node

    def entity_barycenter(self, etype=2, index=np.s_[:]):
        """
        @brief 
        """
        node = self.node
        if etype in {'cell', 2}:
            pass
        elif etype in {'edge', 'face', 1}:
            pass
        elif etype in {'node', 0}:
            bc = node[index]
        else:
            raise ValueError('the entity `{}` is not correct!'.format(entity)) 
        return bc

    def cell_area(self):
        """
        @brief
        """
        return self.h[0]*self.h[1]

    def edge_length(self):
        """
        """
        pass

