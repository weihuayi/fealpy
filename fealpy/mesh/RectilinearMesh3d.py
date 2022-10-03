
import numpy as np


class RectilinearMesh3d():

    def __init__(self, x, y, z):
        self.x = x # shape = (nx+1, )
        self.y = y # shape = (ny+1, )
        self.z = z # shape = (nz+1, )
