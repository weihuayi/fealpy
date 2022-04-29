import jax.numpy as jnp
import numpy as np


class TriangleMesh:
    def __init__(self, node, cell):
        assert cell.shape[-1] == 3

        self.node = node
        NN = node.shape[0]
        self.ds = TriangleMeshDataStructure(NN, cell)

        if node.shape[1] == 2:
            self.meshtype = 'tri'
        elif node.shape[1] == 3:
            self.meshtype = 'stri'

        self.itype = cell.dtype
        self.ftype = node.dtype
        self.p = 1 # 平面三角形

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.meshdata = {}
        

class TriangleMeshDataStructure():
    localEdge = jnp.array([(1, 2), (2, 0), (0, 1)])
    localFace = jnp.array([(1, 2), (2, 0), (0, 1)])
    ccw = jnp.array([0, 1, 2])

    NVC = 3
    NVE = 2
    NVF = 2

    NEC = 3
    NFC = 3

    def __init__(self, NN, cell):
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.itype = cell.dtype
        self.construct()

    def reinit(self, NN, cell):
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.construct()

    def clear(self):
        self.edge = None
        self.edge2cell = None

    def number_of_nodes_of_cells(self):
        return self.NVC

    def number_of_edges_of_cells(self):
        return self.NEC

    def number_of_faces_of_cells(self):
        return self.NEC

    def number_of_vertices_of_cells(self):
        return self.NVC

    def total_edge(self):
        NC = self.NC

        cell = self.cell
        localEdge = self.localEdge

        totalEdge = cell[:, localEdge].reshape(-1, 2)
        return totalEdge

    def local_edge(self):
        return self.localEdge

    def construct(self):
        """ Construct edge and edge2cell from cell
        """
        NC = self.NC
        NEC = self.NEC

        totalEdge = self.total_edge()
        _, i0, j = np.unique(np.sort(totalEdge, axis=-1),
                return_index=True,
                return_inverse=True,
                axis=0)
        NE = i0.shape[0]
        self.NE = NE

        edge2cell = np.zeros((NE, 4), dtype=self.itype)

        i1 = np.zeros(NE, dtype=self.itype)
        i1[j] = np.arange(NEC*NC, dtype=self.itype)

        edge2cell[:, 0] = i0//NEC
        edge2cell[:, 1] = i1//NEC
        edge2cell[:, 2] = i0%NEC
        edge2cell[:, 3] = i1%NEC

        self.edge2cell = jnp.array(edge2cell)
        self.edge = totalEdge[i0, :]

if __name__ == "__main__":

    node = jnp.array([
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1]], dtype=jnp.float64)

    cell = jnp.array([
        [1, 2, 0], 
        [3, 0, 2]], dtype=jnp.int32)


    mesh = TriangleMesh(node, cell)
