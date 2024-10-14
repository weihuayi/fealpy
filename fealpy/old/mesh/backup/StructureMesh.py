
import numpy as np
from numpy.linalg import inv

class UniformStructureMesh:
    """
    @brief 均匀结构网格，每个坐标轴方向都为均匀剖分
    """
    def __init__(self, box, N, itype=np.int_, ftype=np.float64):
        """
        @brief 

        @param[in] box [xmin, xmax] or [xmin, xmax, ymin, ymaxx, ...] 
        @param[in] N 类似数组的有
        """
        assert len(N) == len(box)//2

        self.itype = itype
        self.ftype = ftype 

        self.box = np.array(box, dtype=ftype)
        N = np.array(N, dttype=np.int_) 
        self.GD = len(N)
        self.h = (self.box[1::2] - self.box[0::2])/N
        self.ds = StructureMeshDataStructure(N, itype)

        self.celldata = {}
        self.nodedata = {}
        self.meshdata = {}


    def uniform_refine(self, n=1, returnim=False):
        for i in range(n):
            N = self.ds.N*2
            self.h = (self.box[1::2] - self.box[0::2])/N
            self.ds = StructureMeshDataStructure(N, self.itype)

    def number_of_nodes(self):
        return np.prod(self.ds.N + 1)

    def number_of_cells(self):
        return np.prod(self.ds.N)


class StructureMeshDataStructure():
    def __init__(self, N, itype):
        self.N = N
        self.itype = itype
