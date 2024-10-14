
import numpy as np

from .Mesh3d import Mesh3d
from .StructureMesh3dDataStructure import StructureMesh3dDataStructure


class RectilinearMesh3d(Mesh3d):
    def __init__(self, x, y, z, itype=np.int_):
        self.x = x # shape = (nx+1, )
        self.y = y # shape = (ny+1, )
        self.z = z # shape = (nz+1, )
        nx = len(x) - 1  
        ny = len(y) - 1
        nz = len(z) - 1

        self.ds = StructureMesh3dDataStructure(nx, ny, nz)

        self.itype = itype 
        self.ftype = x.dtype 
        self.meshtype = 'RectilinearMesh3d'

    def geo_dimension(self):
        return 3

    def top_dimension(self):
        return 2

    @property
    def node(self):
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz 
        node = np.zeros((nx+1, ny+1, nz+1, 3), dtype=self.ftype)
        node[..., 0], node[..., 1] = np.mgrid[x, y]
        return node
