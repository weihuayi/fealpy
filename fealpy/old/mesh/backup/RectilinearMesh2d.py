import numpy as np

from .Mesh2d import Mesh2d
from .StructureMesh2dDataStructure import StructureMesh2dDataStructure

class RectilinearMesh2d(Mesh2d):
    """
    @brief 
    """
    def __init__(self, x, y, itype=np.int_):
        """
        """
        self.x = x  # (nx+1, )
        self.y = y  # (ny+1, )

        nx = len(x) - 1  
        ny = len(y) - 1 
        self.ds = StructureMesh2dDataStructure(nx, ny, itype=itype)

        self.itype = itype 
        self.ftype = x.dtype 
        self.meshtype = 'RectilinearMesh2d'

    def geo_dimension(self):
        return 2

    def top_dimension(self):
        return 2

    @property
    def node(self):
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        node = np.zeros((nx+1, ny+1, GD), dtype=self.ftype)
        node[..., 0], node[..., 1] = np.mgrid[x, y]
        return node
