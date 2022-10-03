
import numpy as np
from .StructureMesh3dDataStructure import StructureMesh3dDataStructure

class UniformMesh3d():
    """
    @brief 
    """
    def __init__(self, extent, 
            spacing=(1.0, 1.0, 1.0), 
            origin=(0.0, 0.0, 0.0)
            ):
        self.extent = extent
        self.spacing = spacing 
        self.origin = origin

        nx = extent[1] - extent[0]
        ny = extent[3] - extent[2]
        nz = extent[5] - extent[4]
        self.ds = StructureMesh3dDataStructure(nx, ny, nz)

    @property
    def node(self):
        """
        @brief 获取所有网格节点的坐标
        """
        NN = self.ds.NN
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        extent = self.extent
        hx, hy, hz = self.spacing
        node = np.zeros((NN, 3), dtype=np.float64)
        X, Y, Z = np.mgrid[
                  extent[0]:extent[1]:complex(0, nx + 1),
                  extent[2]:extent[3]:complex(0, ny + 1),
                  extent[4]:extent[5]:complex(0, nz + 1)
                  ]
        node[:, 0] = X.flat
        node[:, 1] = Y.flat
        node[:, 2] = Z.flat

        node[:, 0] *= hx 
        node[:, 1] *= hy
        node[:, 2] *= hz

        return node

    def number_of_nodes(self):
        return self.ds.NN

    def number_of_cells(self):
        return self.ds.NC
