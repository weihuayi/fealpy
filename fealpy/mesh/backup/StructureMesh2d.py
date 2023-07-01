
import numpy as np

from .Mesh2d import Mesh2d
from .StructureMesh2dDataStructure import StructureMesh2dDataStructure

class StructureMesh2d(Mesh2d):
    """
    @brief 二维结构网格
    """
    def __init__(self, node, itype=np.int_):
        """
        """
        self.node = node # (nx+1, ny+1, GD)
        nx = node.shape[0] - 1
        ny = node.shape[1] - 1

        self.ds = StructureMesh2dDataStructure(nx, ny, itype=itype)

        self.itype = itype 
        self.ftype = node.dtype
        self.meshtype = 'StructureMesh2d'

    def geo_dimension(self):
        return self.node[-1] 

    def top_dimension(self):
        return 2
