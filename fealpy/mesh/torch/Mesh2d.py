
from .Mesh import Mesh

class Mesh2d(Mesh):
    def top_dimension(self):
        return 2

class Mesh2dDataStructure():
    """ The topology data structure of mesh 2d
        This is just a abstract class, and you can not use it directly.
    """

    def __init__(self, NN, cell):
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.itype = cell.dtype



