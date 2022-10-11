
import numpy as np


class UniformMesh2d():

    def __init__(self, extent, 
            h=(1.0, 1.0), 
            origin=(0.0, 0.0)
            ):
        self.extent = extent
        self.h = h 
        self.origin = origin

        nx = extent[1] - extent[0]
        ny = extent[3] - extent[2]
        self.ds = StructureMesh3dDataStructure(nx, ny, nz)
