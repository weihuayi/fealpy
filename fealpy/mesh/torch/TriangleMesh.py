import torch

from .Mesh2d import Mesh2d, Mesh2dDataStructure

class TriangleMeshDataStructure(Mesh2dDataStructure):

    localEdge = np.array([(1, 2), (2, 0), (0, 1)])
    localFace = np.array([(1, 2), (2, 0), (0, 1)])
    ccw = np.array([0, 1, 2])

    NVC = 3
    NVE = 2
    NVF = 2

    NEC = 3
    NFC = 3

    localCell = np.array([
        (0, 1, 2),
        (1, 2, 0),
        (2, 0, 1)])

    def __init__(self, NN, cell):
        super().__init__(NN,cell)

class TriangleMesh(Mesh2d):
    def __init__(self, node, cell, itype=torch.uint32, ftype=torch.float64):
        self.itype = itype
        self.ftype = ftype
        self.node = torch.tensor(node, dtype=ftype)
        cell = torch.tensor(cell, dtype=itype)
        NN = len(node)
        self.ds = TriangleMeshDataStructure(len(node), cell)
    
    def cell_area(self):
        p0, p1, p2 = self.node[self.ds.cell[:, 0]], self.node[self.ds.cell[:, 1]], self.node[self.ds.cell[:, 2]]
        v1 = p1 - p0
        v2 = p2 - p0
        cross_product = torch.cross(torch.cat([v1, torch.zeros(v1.shape[0], 1)] ,dim=1),
                                    torch.cat([v2, torch.zeros(v2.shape[0], 1)], dim=1), dim=1)
        areas = 0.5 * cross_product.norm(dim=1)
        return areas
