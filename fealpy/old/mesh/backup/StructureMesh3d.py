
import numpy as np
from .Mesh3d import Mesh3d
from .StructureMesh3dDataStructure import StructureMesh3dDataStructure


class StructureMesh3d(Mesh3d):
    """
    @brief 三维结构网格
    """
    def __init__(self, node, itype=np.int_):
        self.node = node # shpae = (nx+1, ny+1, nz+1, 3)
        nx = node.shape[0] - 1
        ny = node.shape[1] - 1
        nz = node.shape[2] - 1

        self.ds = StructureMesh3dDataStructure(nx, ny, nz, itype=itype)

        self.itype = itype 
        self.ftype = node.dtype
        self.meshtype = 'StructureMesh3d'

    def geo_dimension(self):
        return 3 

    def top_dimension(self):
        return 2
