from ..backend import backend_manager as bm
from .mesh_base import SimplexMesh

class TetrahedronMesh(SimplexMesh): 
    def __init__(self, node, cell):
        super(TetrahedronMesh, self).__init__(TD=3)
        self.node = node
        self.cell = cell

        self.meshtype = 'tet'
        self.p = 1 # linear mesh

        kwargs = {"dtype": self.cell.dtype, } # TODO: 增加 device 参数
        self.localEdge = bm.tensor([
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], **kwargs)
        self.localFace = bm.tensor([
            (1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)], **kwargs)
        self.localCell = bm.tensor([
            (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
            (1, 2, 0, 3), (1, 0, 3, 2), (1, 3, 2, 0),
            (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
            (3, 0, 2, 1), (3, 2, 1, 0), (3, 1, 0, 2)], **kwargs)

        self.ccw = bm.tensor([0, 1, 2, 4], **kwargs)
        self.construct()
        self.OFace = bm.tensor([
            (1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)], **kwargs)
        self.SFace = bm.tensor([
            (1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)], **kwargs)
        self.localFace2edge = bm.tensor([
            (5, 4, 3), (5, 1, 2), (4, 2, 0), (3, 0, 1)], **kwargs)
        self.localEdge2face = bm.tensor(
                [[2, 3], [3, 1], [1, 2], [0, 3], [2, 0], [0, 1]], **kwargs)

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {} 
        self.celldata = {}
        self.meshdata = {}

    ## @ingroup MeshGenerators
    @classmethod
    def from_one_tetrahedron(cls, meshtype='equ'):
        """
        """
        if meshtype == 'equ':
            node = bm.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, bm.sqrt(bm.tensor(3))/2, 0.0],
                [0.5, bm.sqrt(bm.tensor(3))/6, bm.sqrt(bm.tensor(2/3))]], dtype=bm.float64)
        elif meshtype == 'iso':
            node = bm.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int32)
        return cls(node, cell)

    def face_to_edge_sign(self):
        face2edge = self.face_to_edge()
        edge = self.edge
        face = self.face
        NF = len(face2edge)
        NEF = 3
        face2edgeSign = bm.zeros((NF, NEF), dtype=bm.bool_)
        n = [1, 2, 0]
        for i in range(3):
            face2edgeSign[:, i] = (face[:, n[i]] == edge[face2edge[:, i], 0])
        return face2edgeSign
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_box(cls, box=[0, 1, 0, 1, 0, 1], nx=10, ny=10, nz=10, threshold=None):
        """
        Generate a tetrahedral mesh for a box domain.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param nz Number of divisions along the z-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TetrahedronMesh instance
        """
        NN = (nx+1)*(ny+1)*(nz+1)
        NC = nx*ny*nz
        node = bm.zeros((NN, 3), dtype=bm.float64)
        x = bm.linspace(box[0], box[1], nx+1)
        y = bm.linspace(box[2], box[3], ny+1)
        z = bm.linspace(box[4], box[5], nz+1)
        X, Y, Z = bm.meshgrid(x, y, z, indexing='ij')
 
        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), axis=1)

        idx = bm.arange(NN).reshape(nx+1, ny+1, nz+1)
        c = idx[:-1, :-1, :-1]

        nyz = (ny + 1)*(nz + 1)
        cell0 = idx[:-1, :-1, :-1] 
        cell1 = cell0 + nyz
        cell2 = cell1 + nz + 1
        cell3 = cell0 + nz + 1
        cell4 = cell0 + 1
        cell5 = cell4 + nyz
        cell6 = cell5 + nz + 1
        cell7 = cell4 + nz + 1
        cell = bm.concatenate((cell0.reshape(-1, 1), cell1.reshape(-1, 1),
            cell2.reshape(-1, 1), cell3.reshape(-1, 1), cell4.reshape(-1, 1),
            cell5.reshape(-1, 1), cell6.reshape(-1, 1), cell7.reshape(-1, 1)),
            axis = 1)

        localCell = bm.tensor([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=bm.int32)
        cell = cell[:, localCell].reshape(-1, 4)

        if threshold is not None:
            NN = len(node)
            bc = bm.sum(node[cell, :], axis=1)/cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = bm.zeros(NN, dtype=bm.bool_)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]
        mesh = TetrahedronMesh(node, cell)
        import ipdb
        ipdb.set_trace()
        bdface = mesh.boundary_face_index()
        f2n = mesh.face_unit_normal()[bdface]
        isLeftBd   = bm.abs(f2n[:, 0]+1)<1e-14
        isRightBd  = bm.abs(f2n[:, 0]-1)<1e-14
        isFrontBd  = bm.abs(f2n[:, 1]+1)<1e-14
        isBackBd   = bm.abs(f2n[:, 1]-1)<1e-14
        isBottomBd = bm.abs(f2n[:, 2]+1)<1e-14
        isUpBd     = bm.abs(f2n[:, 2]-1)<1e-14
        mesh.meshdata["leftface"]   = bdface[isLeftBd]
        mesh.meshdata["rightface"]  = bdface[isRightBd]
        mesh.meshdata["frontface"]  = bdface[isFrontBd]
        mesh.meshdata["backface"]   = bdface[isBackBd]
        mesh.meshdata["upface"]     = bdface[isUpBd]
        mesh.meshdata["bottomface"] = bdface[isBottomBd]
        return mesh



