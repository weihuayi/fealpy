import numpy as np
from .Mesh2d import Mesh2d, Mesh2dDataStructure


class QuadtreeMeshDataStructure(Mesh2dDataStructure):
    """
    Notice that vertex order of cell in QuadMesh is z-order as following:

        c_yc_x

        0: 00
        1: 01
        2: 10
        3: 11

        2-------------3
        |             |
        |             |
        |             |
        |             |
        0-------------1

    cell2cell <===> NO(k, f) = k'
    r \in {0, 1}
    f' \in {0, 1, 2, 3}
    NF(k, f) = 4r + f'

    if f is a boundary,  NF(k, f) = 4r+f

    for k-th octree, is f < f'
    """
    localEdge = np.array([(0, 2), (1, 3), (0, 1), (2, 3)])
    ccw = np.array([0, 1, 3, 2], dtype=np.int8)
    V = 4
    E = 4
    F = 1

    def __init__(self, NN, cell):
        super(QuadtreeMeshDataStructure, self).__init__(NN, cell)


class QuadtreeMesh(Mesh2d):
    def __init__(self, node, cell):
        self.node = node
        NN = node.shape[0]
        self.ds = QuadtreeMeshDataStructure(NN, cell)

        self.meshtype = 'quadtreemesh'

        self.itype = cell.dtype
        self.ftype = node.dtype

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}

    def uniform_refine(self, n=1):
        for i in range(n):
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NC = self.number_of_cells()

            # Find the cutted edge 
            cell2edge = self.ds.cell_to_edge()
            edgeCenter = self.entity_barycenter('edge')
            cellCenter = self.entity_barycenter('cell')

            edge2center = np.arange(NN, NN+NE)

            cell = self.ds.cell
            cp = [cell[:, i].reshape(-1, 1) for i in range(4)]
            ep = [edge2center[cell2edge[:, i]].reshape(-1, 1) for i in range(4)]
            cc = np.arange(NN + NE, NN + NE + NC).reshape(-1, 1)

            cell = np.zeros((4*NC, 4), dtype=np.int)
            cell[0::4, :] = np.r_['1', cp[0], ep[2], ep[0], cc]
            cell[1::4, :] = np.r_['1', ep[2], cp[1], cc, ep[1]]
            cell[2::4, :] = np.r_['1', ep[0], cc, cp[2], ep[3]]
            cell[3::4, :] = np.r_['1', cc, ep[1], ep[3], cp[3]]

            self.node = np.r_['0', self.node, edgeCenter, cellCenter]
            self.ds.reinit(NN + NE + NC, cell)

    def bc_to_point(self, bc):
        """
        重心坐标的顺序是逆时针方向，而 QuadMesh 单元顶点的编号满足 z-order
 
        2-------------3
        |             |
        |             |
        |             |
        |             |
        0-------------1
        """
        node = self.node
        cell = self.ds.cell
        p = np.einsum('...j, ijk->...ik', bc, node[cell[:, self.ccw]])
        return p


class QuadtreeForest():
    """

    leaf node:
    interior node:
    siblings:
    descendants:
    ancestor:
    subtree:
    level: The depth of a node from the root.

    linear octrees:
        + It has lower storage costs than other representations.
        + The other representations use pointers, which add synchronization 
          and communication overhead for parallel implementations.

    Morton encoding:
    anchor: Any octant in the domain can be uniquely identified
            by specifying one of its vertices

    (4, 2) --> (0100, 0010)--> from right to left interleave 00011000
    """
    def __init__(self, mesh, maxdepth=32):
        self.mesh = mesh
        self.maxdepth = maxdepth
        NC = self.mesh.number_of_cells()
        self.levels = np.empty(NC, dtype=np.ndarray)
        self.forest = np.empty(NC, dtype=np.ndarray)

        self.forest.fill(np.zeros(1, dtype=np.uint64))
        self.levels.fill(np.zeros(1, dtype=np.uint8))

    def number_of_trees(self):
        return self.forest.shape[0]

    def uniform_refine(self, n=1):
        NT = self.number_of_trees();
        for i in range(n):
            for j in range(NT):
                NL = len(self.forest[j])
                level = np.repeat(self.levels[j], 4)
                tree = np.repeat(self.forest[j], 4)
                child = np.tile(np.array([0, 2, 1, 3], dtype=np.uint64), NL)
                self.forest[j] = tree + np.left_shift(child, 2*level)
                self.levels[j] = level + np.uint8(1)

    def octant(self):
        NT = self.number_of_trees();
        maxdepth = self.maxdepth
        for j in range(NT):
            tree = self.forest[j]
            NL = len(tree)
            a = np.unpackbits(tree.view(np.uint8)).reshape(-1, 64)
            print(a)
            x = np.zeros((NL, 32), dtype=np.uint8)
            y = np.zeros((NL, 32), dtype=np.uint8)
            x = a[:, 0::2]
            y = a[:, 1::2]
            print(a)
            print(x)
            print(y)
            x = np.packbits(x.flat).view(np.uint32)
            y = np.packbits(y.flat).view(np.uint32)


    def print(self):
        NT = self.number_of_trees()
        for j in range(NT):
            print("The {0}-th tree:\n".format(j))
            print("levels:\n", self.levels[j])
            print([bin(x)[2:].zfill(64) for x in self.forest[j]])

    def add_plot(self, plt):
        mesh = self.mesh
        maxdepth = self.maxdepth

        qmesh = QuadtreeMesh(mesh.node.copy(), mesh.ds.cell.copy())
        qmesh.uniform_refine(maxdepth)
        qmesh.node *= 2**maxdepth
        fig = plt.figure()
        axes = fig.gca()
        qmesh.add_plot(axes, showaxis=True, cellcolor='lightgray', edgecolor='gray', linewidths=1)
        plt.xticks(np.arange(2**maxdepth+1))
        plt.yticks(np.arange(2**maxdepth+1))


