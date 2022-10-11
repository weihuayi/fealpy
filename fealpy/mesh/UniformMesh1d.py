
import numpy as np

class UniformMesh3d():
    """
    @brief 
    """
    def __init__(self, extent, 
            h=1.0, 
            origin=0.0
            ):
        self.extent = extent
        self.h = h 
        self.origin = origin

        nx = extent[1] - extent[0]
