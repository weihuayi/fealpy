import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from scipy.sparse import triu, tril, find, hstack
from .mesh_tools import unique_row
from .Mesh3d import Mesh3d, Mesh3dDataStructure 

class HexahedronMeshDataStructure(Mesh3dDataStructure):

    # The following local data structure should be class properties
    localEdge = np.array([
        (0, 1), (1, 2), (2, 3), (0, 3),
        (0, 4), (1, 5), (2, 6), (3, 7),
        (4, 5), (5, 6), (6, 7), (4, 7)])
    localFace = np.array([
        (0, 3, 2, 1), (4, 5, 6, 7), # bottom and top faces
        (0, 4, 7, 3), (1, 2, 6, 5), # left and right faces  
        (0, 1, 5, 4), (2, 3, 7, 6)])# front and back faces
    localFace2edge = np.array([
        (3,  2, 1, 0), (8, 9, 10, 11),
        (4, 11, 7, 3), (1, 6,  9,  5),
        (0,  5, 8, 4), (2, 7, 10,  6)])
    NVC = 8
    NEC = 12
    NFC = 6
    NVF = 4
    NEF = 4

    def __init__(self, NN, cell):
        super().__init__(NN, cell)

        
    def face_to_edge_sign(self):
        face2edge = self.face_to_edge()
        edge = self.edge
        face2edgeSign = np.zeros((NF, 4), dtype=np.bool)
        for i in range(4):
            face2edgeSign[:, i] = (face[:, i] == edge[face2edge[:, i], 0])
        return face2edgeSign

class HexahedronMesh(Mesh3d):
    """
    @brief 非结构六面体网格数据结构对象
    """
    def __init__(self, node, cell):
        self.node = node
        NN = node.shape[0]
        self.ds = HexahedronMeshDataStructure(NN, cell)

        self.meshtype = 'hex'

        self.itype = cell.dtype
        self.ftype = node.dtype

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {} 
        self.meshdata = {}

    def integrator(self, q, etype='cell'):
        """
        @brief 获取不同维度网格实体上的积分公式 
        """
        qf = GaussLegendreQuadrature(q)
        if etype in {'cell', 3}:
            return TensorProductQuadrature((qf, qf, qf)) 
        elif etype in {'face', 2}:
            return TensorProductQuadrature((qf, qf)) 
        elif etype in {'edge', 1}:
            return qf 

    def bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把重心坐标积分点变换为网格实体上的笛卡尔坐标点
        """
        node = self.entity('node')
        if isinstance(bc, tuple):
            if len(bc) == 2:
                face = self.entity('face')[index]
                bc0 = bc[0] # (NQ0, 2)
                bc1 = bc[1] # (NQ1, 2)
                bc = np.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 4) # (NQ0, NQ1, 2, 2)
                p = np.einsum('...j, cjk->...ck', bc, node[face[:, [0, 3, 1, 2]]]) # (NQ, NC, 2)
            elif len(bc) == 3:
                cell = self.entity('cell')[index]
                bc0 = bc[0] # (NQ0, 2)
                bc1 = bc[1] # (NQ1, 2)
                bc2 = bc[2] # (NQ1, 2)
                bc = np.einsum('im, jn, kl->ijkmnl', bc0, bc1, bc2).reshape(-1, 8) # (NQ0, NQ1, NQ2, 2, 2, 2)
                p = np.einsum('...j, cjk->...ck', bc, node[cell[:, [0, 4, 3, 7, 1, 5, 2, 6]]]) # (NQ, NC, 3)
        else:
            edge = self.entity('edge')[index]
            p = np.einsum('...j, ejk->...ek', bc, node[edge]) # (NQ, NE, 3)
        return p 

    def edge_bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把重心坐标积分点变换为网格边上的笛卡尔坐标点
        """
        node = self.node
        entity = self.entity('edge')[index]
        p = np.einsum('...j, ijk->...ik', bc, node[entity])
        return p

    def face_bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把重心坐标积分点变换为网格面上的笛卡尔坐标点
        """
        assert len(bc) == 2
        node = self.entity('node')
        face = self.entity('face')[index]
        bc0 = bc[0] # (NQ0, 2)
        bc1 = bc[1] # (NQ1, 2)
        bc = np.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 4) # (NQ0, NQ1, 2, 2)
        p = np.einsum('...j, cjk->...ck', bc, node[face[:, [0, 3, 1, 2]]]) # (NQ, NC, 2)
        return p

    def cell_bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把重心坐标积分点变换为网格单元上的笛卡尔坐标点
        """
        assert len(bc) == 3
        node = self.entity('node')
        cell = self.entity('cell')[index]
        bc0 = bc[0] # (NQ0, 2)
        bc1 = bc[1] # (NQ1, 2)
        bc2 = bc[2] # (NQ2, 2)
        bc = np.einsum('im, jn, kl->ijkmnl', bc0, bc1, bc2).reshape(-1, 8) # (NQ0, NQ1, NQ2, 2, 2, 2)
        p = np.einsum('...j, cjk->...ck', bc, node[cell[:, [0, 4, 3, 7, 1, 5, 2, 6]]]) # (NQ, NC, 3)
        return p

    def uniform_refine(self, n=1):
        """
        @brief 一致加密六面体网格 
        """
        for i in range(n):
            pass


    def cell_volume(self):
        pass

    def face_area(self):
        pass

    def jacobi_at_corner(self):
        pass

    def cell_quality(self):
        pass

    def print(self):
        print("Point:\n", self.node)
        print("Cell:\n", self.ds.cell)
        print("Edge:\n", self.ds.edge)
        print("Face:\n", self.ds.face)
        print("Face2cell:\n", self.ds.face2cell)


