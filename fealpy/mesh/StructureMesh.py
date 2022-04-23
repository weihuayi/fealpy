
import numpy as np
from numpy.linalg import inv

class StructureMesh:
    def __init__(self, box, N, dft=None):
        self.box = box
        self.N = N
        self.GD = box.shape[0] 

        self.ftype = np.float64
        self.itype = np.int_
