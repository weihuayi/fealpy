
import numpy as np

from .StructureMesh3dDataStructure import StructureMesh3dDataStructure


class RectilinearMesh3d():
    def __init__(self, x, y, z):
        self.x = x # shape = (nx+1, )
        self.y = y # shape = (ny+1, )
        self.z = z # shape = (nz+1, )
        nx = len(x) - 1  
        ny = len(y) - 1
        nz = len(z) - 1
        self.ds = StructureMesh3dDataStructure(nx, ny, nz)
        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {}

    def vtk_cell_type(self):
        VTK_HEXAHEDRON = 12
        return VTK_HEXAHEDRON

    def to_vtk_file(self, filename, celldata=None, nodedata=None):
        """

        """
        from pyevtk.hl import gridToVTK

        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz

        gridToVTK(filename, x, y, z, cellData=celldata, pointData=nodedata)

        return filename 

    @property
    def node(self):
        NN = self.ds.NN
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        node = np.zeros((NN, 3), dtype=np.float)
        X, Y, Z = np.mgrid[
                  box[0]:box[1]:complex(0, nx + 1),
                  box[2]:box[3]:complex(0, ny + 1),
                  box[4]:box[5]:complex(0, nz + 1)
                  ]
        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()
        node[:, 2] = Z.flatten()

        return node

    def number_of_nodes(self):
        return self.ds.NN

    def number_of_cells(self):
        return self.ds.NC

