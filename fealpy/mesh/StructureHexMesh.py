import numpy as np
from .Mesh3d import Mesh3d

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse import spdiags, eye, tril, triu, diags, kron


class StructureHexMesh(Mesh3d):
    def __init__(self, box, nx, ny, nz, itype=np.int_, ftype=np.float64):
        self.itype = itype
        self.ftype = ftype
        self.box = box
        self.hx = (box[1] - box[0]) / nx
        self.hy = (box[3] - box[2]) / ny
        self.hz = (box[5] - box[4]) / nz
        self.ds = StructureHexMeshDataStructure(nx, ny, nz)

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {}
        self.meshdata = {}

    def multi_index(self):
        NN = self.ds.NN
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        i, j, k = np.mgrid[0:nx + 1, 0:ny + 1, 0:nz + 1]
        index = np.zeros((NN, 3), dtype=self.itype)
        index[:, 0] = i.flat
        index[:, 1] = j.flat
        index[:, 2] = k.flat
        return index

    def uniform_refine(self, n=1, returnim=False):
        """
        2022.5.8 新加
        """
        if returnim:
            nodeImatrix = []

        for i in range(n):
            nx = 2 * self.ds.nx
            ny = 2 * self.ds.ny
            nz = 2 * self.ds.nz
            self.ds = StructureHexMeshDataStructure(nx, ny, nz)
            self.hx = (self.box[1] - self.box[0]) / nx
            self.hy = (self.box[3] - self.box[2]) / ny
            self.hz = (self.box[5] - self.box[4]) / nz

            if returnim:
                A = self.interpolation_matrix()
                nodeImatrix.append(A)

        if returnim:
            return nodeImatrix

    def interpolation_matrix(self):
        """
        @brief 加密一次的插值矩阵
        """

        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz

        nxH = nx // 2
        nyH = ny // 2
        nzH = nz // 2

        NNH = (nx // 2 + 1) * (ny // 2 + 1) * (nz // 2 + 1)
        NNh = self.number_of_nodes()

        I = np.arange(NNh).reshape(nx + 1, ny + 1, nz + 1)
        J = np.arange(NNH).reshape(nx // 2 + 1, ny // 2 + 1, nz // 2 + 1)

        ## (2i, 2j, 2k)
        I1 = I[::2, ::2, ::2].flat
        J1 = J.flat
        data = np.broadcast_to(1, (len(I1),))
        A = coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        ## (2i+1, 2j, 2k)
        I1 = I[1::2, ::2, ::2].flat
        J1 = J[:-1, :, :].flat  # (i,j,k)
        data = np.broadcast_to(1 / 2, (len(I1),))
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[1:, :, :].flat  # (i+1,j,k)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        ## (2i, 2j+1, 2k)
        I1 = I[::2, 1::2, ::2].flat
        J1 = J[:, :-1, :].flat  # (i,j,k)
        data = np.broadcast_to(1 / 2, (len(I1),))
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[:, 1:, :].flat  # (i,j+1,k)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        ## (2i, 2j, 2k+1)
        I1 = I[::2, ::2, 1::2].flat
        J1 = J[:, :, :-1].flat  # (i,j,k)
        data = np.broadcast_to(1 / 2, (len(I1),))
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[:, :, 1:].flat  # (i,j,k+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        ## (2i+1, 2j+1, 2k)
        I1 = I[1::2, 1::2, ::2].flat
        J1 = J[:-1, :-1, :].flat  # (i,j,k)
        data = np.broadcast_to(1 / 4, (len(I1),))
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[1:, :-1, :].flat  # (j+1,j,k)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[:-1, 1:, :].flat  # (i,j+1,k)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[1:, 1:, :].flat  # (i+1,j+1,k)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        ## (2i, 2j+1, 2k+1)
        I1 = I[::2, 1::2, 1::2].flat
        J1 = J[:, :-1, :-1].flat  # (i,j,k)
        data = np.broadcast_to(1 / 4, (len(I1),))
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[:, 1:, :-1].flat  # (i,j+1,k)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[:, :-1, 1:].flat  # (i,j,k+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[:, 1:, 1:].flat  # (i,j+1,k+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        ## (2i+1, 2j, 2k+1)
        I1 = I[1::2, ::2, 1::2].flat
        J1 = J[:-1, :, :-1].flat  # (i,j,k)
        data = np.broadcast_to(1 / 4, (len(I1),))
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[1:, :, :-1].flat  # (i+1,j,k)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[:-1, :, 1:].flat  # (i,j,k+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[1:, :, 1:].flat  # (i+1,j,k+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        ## (2i+1, 2j+1, 2k+1)
        I1 = I[1::2, 1::2, 1::2].flat
        J1 = J[:-1, :-1, :-1].flat  # (i,j,k)
        data = np.broadcast_to(1 / 8, (len(I1),))
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[1:, :-1, :-1].flat  # (i+1,j,k)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[:-1, 1:, :-1].flat  # (i,j+1,k)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[1:, 1:, :-1].flat  # (i+1,j+1,k)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[:-1, :-1, 1:].flat  # (i,j,k+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[1:, :-1, 1:].flat  # (i+1,j,k+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[:-1, 1:, 1:].flat  # (i,j+1,k+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[1:, 1:, 1:].flat  # (i+1,j+1,k+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        return A

    def vtk_cell_type(self):
        VTK_HEXAHEDRON = 12
        return VTK_HEXAHEDRON

    def to_vtk_file(self, filename, celldata=None, nodedata=None):
        """


        """
        from pyevtk.hl import gridToVTK

        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        box = self.box

        x = np.linspace(box[0], box[1], nx+1)
        y = np.linspace(box[2], box[3], ny+1)
        z = np.linspace(box[4], box[5], nz+1)
        gridToVTK(filename, x, y, z, cellData=celldata, pointData=nodedata)

        return filename 


    @property
    def node(self):
        NN = self.ds.NN
        box = self.box
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        node = np.zeros((NN, 3), dtype=np.float)
        X, Y, Z = np.mgrid[
                  box[0]:box[1]:complex(0, nx + 1),
                  box[2]:box[3]:complex(0, ny + 1),
                  box[4]:box[5]:complex(0, nz + 1)
                  ]
        node[:, 0] = X.flatten()
        node[:, 1] = Y.flatten()
        node[:, 2] = Z.flatten()

        return node

    def number_of_nodes(self):
        return self.ds.NN

    def number_of_cells(self):
        return self.ds.NC

    def laplace_operator(self):
        """
        @brief 构造笛卡尔网格上的 Laplace 离散算子，其中 x, y, z
        三个方向都是均匀剖分，但各自步长可以不一样
        @todo 处理带系数的情形
        """

        n0 = self.ds.nx + 1
        n1 = self.ds.ny + 1
        n2 = self.ds.nz + 1

        cx = 1 / (self.hx ** 2)
        cy = 1 / (self.hy ** 2)
        cz = 1 / (self.hz ** 2)

        NN = self.number_of_nodes()
        k = np.arange(NN).reshape(n0, n1, n2)

        A = diags([2 * (cx + cy + cz)], [0], shape=(NN, NN), format='coo')

        val = np.broadcast_to(-cx, (NN - n1 * n2,))
        I = k[1:, :, :].flat
        J = k[0:-1, :, :].flat
        A += coo_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += coo_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-cy, (NN - n0 * n2,))
        I = k[:, 1:, :].flat
        J = k[:, 0:-1, :].flat
        A += coo_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += coo_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-cz, (NN - n0 * n1,))
        I = k[:, :, 1:].flat
        J = k[:, :, 0:-1].flat
        A += coo_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += coo_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A.tocsr()

    def function(self, etype='node', dtype=None):
        """
        @brief 返回定义在节点、网格边、网格面、或网格单元上离散函数（数组），元素取值为0

        @todo 明确需要定义的函数的实体集合
        """

        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz

        dtype = self.ftype if dtype is None else dtype

        if etype in {'node', 0}:
            uh = np.zeros((nx + 1, ny + 1, nz + 1), dtype=dtype)
        elif etype in {'edge', 1}:
            ex = np.zeros((nx, ny + 1, nz + 1), dtype=dtype)
            ey = np.zeros((nx + 1, ny, nz + 1), dtype=dtype)
            ez = np.zeros((nx + 1, ny + 1, nz), dtype=dtype)
            uh = (ex, ey, ez)
        elif etype in {'edgex'}:
            uh = np.zeros((nx, ny + 1, nz + 1), dtype=dtype)
        elif etype in {'edgey'}:
            uh = np.zeros((nx + 1, ny, nz + 1), dtype=dtype)
        elif etype in {'edgez'}:
            uh = np.zeros((nx + 1, ny + 1, nz), dtype=dtype)
        elif etype in {'face', 2}:
            fx = np.zeros((nx + 1, ny, nz), dtype=dtype)
            fy = np.zeros((nx, ny + 1, nz), dtype=dtype)
            fz = np.zeros((nx, ny, nz + 1), dtype=dtype)
            uh = (fx, fy, fz)
        elif etype in {'facex'}:
            uh = np.zeros((nx + 1, ny, nz), dtype=dtype)
        elif etype in {'facey'}:
            uh = np.zeros((nx, ny + 1, nz), dtype=dtype)
        elif etype in {'facez'}:
            uh = np.zeros((nx, ny, nz + 1), dtype=dtype)
        elif etype in {'cell', 3}:
            uh = np.zeros((nx, ny, nz), dtype=dtype)
        return uh

    def interpolation(self, f, intertype='node'):
        """
        @brief 把一个已知函数插值到网格节点上或者单元上
        """
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        node = self.node
        if intertype == 'node':
            F = f(node)
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx + 1, ny + 1, nz + 1) + shape
            F = F.reshape(shape)

        elif intertype == 'edge':
            ec = self.entity_barycenter('edge')
            F = f(ec)
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]

            isXDEdge = self.ds.x_direction_edge_flag()
            shape = (nx, ny + 1, nz + 1) + shape
            XF = F[isXDEdge].reshape(shape)

            isYDEdge = self.ds.y_direction_edge_flag()
            shape = (nx + 1, ny, nz + 1) + shape
            YF = F[isYDEdge].reshape(shape)

            isZDEdge = self.ds.z_direction_edge_flag()
            shape = (nx + 1, ny + 1, nz) + shape
            ZF = F[isZDEdge].reshape(shape)
            F = (XF, YF, ZF)

        elif intertype == 'edgex':
            isXDEdge = self.ds.x_direction_edge_flag()
            ec = self.entity_barycenter('edge')
            F = f(ec[isXDEdge])
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx, ny + 1, nz + 1) + shape
            F = F.reshape(shape)
        elif intertype == 'edgey':
            isYDEdge = self.ds.y_direction_edge_flag()
            ec = self.entity_barycenter('edge')
            F = f(ec[isYDEdge])
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx + 1, ny, nz + 1) + shape
            F = F.reshape(shape)
        elif intertype == 'edgez':
            isZDEdge = self.ds.z_direction_edge_flag()
            ec = self.entity_barycenter('edge')
            F = f(ec[isZDEdge])
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx + 1, ny + 1, nz) + shape
            F = F.reshape(shape)

        elif intertype == 'face':
            fc = self.entity_barycenter('face')
            F = f(fc)
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]

            isXDFace = self.ds.x_direction_face_flag()
            shape = (nx + 1, ny, nz) + shape
            XF = F[isXDFace].reshape(shape)

            isYDFace = self.ds.y_direction_face_flag()
            shape = (nx, ny + 1, nz) + shape
            YF = F[isYDFace].reshape(shape)

            isZDFace = self.ds.z_direction_face_flag()
            shape = (nx, ny, nz + 1) + shape
            ZF = F[isZDFace].reshape(shape)
            F = (XF, YF, ZF)

        elif intertype == 'facex':
            isXDFace = self.ds.x_direction_face_flag()
            fc = self.entity_barycenter('face')
            F = f(fc[isXDFace])
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx + 1, ny, nz) + shape
            F = F.reshape(shape)
        elif intertype == 'facey':
            isYDFace = self.ds.y_direction_face_flag()
            fc = self.entity_barycenter('face')
            F = f(fc[isYDFace])
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx, ny + 1, nz) + shape
            F = F.reshape(shape)
        elif intertype == 'facez':
            isZDFace = self.ds.z_direction_face_flag()
            fc = self.entity_barycenter('face')
            F = f(fc[isZDFace])
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx, ny, nz + 1) + shape
            F = F.reshape(shape)

        elif intertype == 'cell':
            bc = self.entity_barycenter('cell')
            F = f(bc)
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx, ny, nz) + shape
            F = F.reshape(shape)
        return F

    def data_edge_to_node(self, Ex, Ey, Ez):
        """
        @brief 把定义在边上的数组转换到节点上
        """
        dx = self.function(etype='node')  # (nx+1, ny+1, nz+1)
        dy = self.function(etype='node')
        dz = self.function(etype='node')

        dx[0:-1, :, :] = Ex
        dx[-1, :, :] = Ex[-1, :, :]
        dx[1:-1, :, :] += Ex[1:, :, :]
        dx[1:-1, :, :] /= 2.0

        dy[:, 0:-1, :] = Ey
        dy[:, -1, :] = Ey[:, -1, :]
        dy[:, 1:-1, :] += Ey[:, 1:, :]
        dy[:, 1:-1, :] /= 2.0

        dz[:, :, 0:-1] = Ez
        dz[:, :, -1] = Ez[:, :, -1]
        dz[:, :, 1:-1] += Ez[:, :, 1:]
        dz[:, :, 1:-1] /= 2.0

        NN = len(dx.flat)
        data = np.zeros((NN, 3), dtype=Ex.dtype)
        data[:, 0] = dx.flat
        data[:, 1] = dy.flat
        data[:, 2] = dz.flat

        return data

    def data_edge_to_cell(self, Ex, Ey, Ez):
        """
        @brief 把定义在边上的数组转换到单元上
        """
        dx = self.function(etype='cell')
        dy = self.function(etype='cell')
        dz = self.function(etype='cell')

        dx[:] = (Ex[:, :-1, :-1] + Ex[:, :-1, 1:] + Ex[:, 1:, :-1] + Ex[:, 1:, 1:])/4.0
        dy[:] = (Ey[:-1, :, :-1] + Ey[1:, :, :-1] + Ey[:-1, :, 1:] + Ey[1:, :, 1:])/4.0
        dz[:] = (Ez[:-1, :-1, :] + Ez[1:, :-1, :] + Ez[:-1, 1:, :] + Ez[1:, 1:, :])/4.0

        return dx, dy, dz

         



class StructureHexMeshDataStructure():
    # The following local data structure should be class properties
    localEdge = np.array([
        (0, 1), (3, 2), (5, 4), (6, 7),
        (2, 0), (1, 3), (4, 6), (7, 5),
        (0, 4), (5, 1), (6, 2), (3, 7)])
    localFace = np.array([
        (0, 2, 6, 4), (1, 5, 7, 3),  # bottom and top faces
        (0, 4, 5, 1), (2, 3, 7, 6),  # left and right faces
        (0, 1, 3, 2), (4, 6, 7, 5)])  # front and back faces
    localFace2edge = np.array([
        (4, 10, 6, 8), (5, 9, 7, 11),
        (0, 8, 2, 9), (1, 11, 3, 10),
        (0, 5, 1, 2), (2, 6, 3, 7)])
    # localEdge = np.array([
    #     (0, 1), (1, 2), (2, 3), (3, 0),
    #     (0, 4), (1, 5), (2, 6), (3, 7),
    #     (4, 5), (5, 6), (6, 7), (7, 4)])
    # localFace = np.array([
    #     (0, 3, 2, 1), (4, 5, 6, 7),  # bottom and top faces
    #     (0, 4, 7, 3), (1, 2, 6, 5),  # left and right faces
    #     (0, 1, 5, 4), (2, 3, 7, 6)])  # front and back faces
    # localFace2edge = np.array([
    #     (0, 1, 2, 3), (8, 9, 10, 11),
    #     (4, 11, 7, 3), (1, 6, 9, 5),
    #     (0, 5, 8, 4), (2, 7, 10, 6)])
    V = 8
    E = 12
    F = 6

    def __init__(self, nx, ny, nz):
        self.nx = nx
        self.ny = ny
        self.nz = nz

        self.NN = (nx + 1) * (ny + 1) * (nz + 1)
        self.NE = (nx + 1) * (ny + 1) * nz + (nx + 1) * ny * (nz + 1) + nx * (ny + 1) * (nz + 1)
        self.NF = nx * ny * (nz + 1) + nx * (ny + 1) * nz + (nx + 1) * ny * nz
        self.NC = nx * ny * nz

    @property
    def cell(self):
        NN = self.NN
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NC = self.NC
        idx = np.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        c = idx[:-1, :-1, :-1]

        cell = np.zeros((NC, 8), dtype=np.int)
        nyz = (ny + 1) * (nz + 1)
        cell[:, 0] = c.flatten()
        cell[:, 1] = cell[:, 0] + nyz
        cell[:, 2] = cell[:, 1] + nz + 1
        cell[:, 3] = cell[:, 0] + nz + 1
        cell[:, 4] = cell[:, 0] + 1
        cell[:, 5] = cell[:, 4] + nyz
        cell[:, 6] = cell[:, 5] + nz + 1
        cell[:, 7] = cell[:, 4] + nz + 1
        return cell

    @property
    def face(self):
        NN = self.NN
        NF = self.NF

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        face = np.zeros((NF, 4), dtype=np.int)
        # NF0 = 0
        # NF1 = (nx + 1) * ny * nz
        # c = idx[:, :-1, :-1]
        # face[NF0:NF1, 0] = c.flatten()
        # face[NF0:NF1, 1] = face[NF0:NF1, 0] + nz + 1
        # face[NF0:NF1, 2] = face[NF0:NF1, 1] + 1
        # face[NF0:NF1, 3] = face[NF0:NF1, 0] + 1
        # face[0:ny * nz, :] = face[0:ny * nz, [0, 3, 2, 1]]

        NF0 = 0
        NF1 = nx * ny * (nz + 1)
        c = np.transpose(idx, (0, 1, 2))[:-1, :-1, :]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
        face[NF0:NF1, 2] = face[NF0:NF1, 1] + nz + 1
        face[NF0:NF1, 3] = face[NF0:NF1, 0] + nz + 1
        face[0:nx * ny, :] = face[0:nx * ny, [0, 3, 2, 1]]

        # NF0 = NF1
        # NF1 += nx * (ny + 1) * nz
        # c = np.transpose(idx, (1, 2, 0))[:, :-1, :-1]
        # face[NF0:NF1, 0] = c.flatten()
        # face[NF0:NF1, 1] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
        # face[NF0:NF1, 2] = face[NF0:NF1, 1] + 1
        # face[NF0:NF1, 3] = face[NF0:NF1, 0] + 1
        # face[(NF1 - nx * nz):NF1, :] = face[(NF1 - nx * nz):NF1, [1, 0, 3, 2]]

        NF0 = NF1
        NF1 += nx * (ny + 1) * nz
        c = np.transpose(idx, (0, 1, 2))[:-1, :, :-1]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
        face[NF0:NF1, 2] = face[NF0:NF1, 1] + 1
        face[NF0:NF1, 3] = face[NF0:NF1, 0] + 1
        face[(NF1 - nx * ny):NF1, :] = face[(NF1 - nx * ny):NF1, [1, 0, 3, 2]]

        # NF0 = NF1
        # NF1 += (nz + 1) * nx * ny
        # c = np.transpose(idx, (2, 0, 1))[:, :-1, :-1]
        # face[NF0:NF1, 0] = c.flatten()
        # face[NF0:NF1, 1] = face[NF0:NF1, 0] + (ny + 1) * (nz + 1)
        # face[NF0:NF1, 2] = face[NF0:NF1, 1] + nz + 1
        # face[NF0:NF1, 3] = face[NF0:NF1, 0] + nz + 1
        # face[NF0:NF0 + nx * ny, :] = face[NF0:NF0 + nx * ny, [0, 3, 2, 1]]

        NF0 = NF1
        NF1 += (nx + 1) * ny * nz
        c = idx[:, :-1, :-1]
        face[NF0:NF1, 0] = c.flatten()
        face[NF0:NF1, 1] = face[NF0:NF1, 0] + nz + 1
        face[NF0:NF1, 2] = face[NF0:NF1, 1] + 1
        face[NF0:NF1, 3] = face[NF0:NF1, 0] + 1
        face[NF0:NF0 + ny * nz, :] = face[NF0:NF0 + ny * nz, [0, 3, 2, 1]]
        return face

    @property
    def face2cell(self):
        NN = self.NN
        NF = self.NF
        NC = self.NC

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NC).reshape(nx, ny, nz)
        face2cell = np.zeros((NF, 4), dtype=np.int)
        # x direction
        NF0 = 0
        NF1 = ny * nz
        face2cell[NF0:NF1, 0] = idx[0].flatten()
        face2cell[NF0:NF1, 1] = idx[0].flatten()
        face2cell[NF0:NF1, 2:4] = 2

        NF0 = NF1
        NF1 += nx * ny * nz
        face2cell[NF0:NF1, 0] = idx.flatten()
        face2cell[NF0:NF1, 2] = 3
        face2cell[NF0:NF1 - ny * nz, 1] = idx[1:].flatten()
        face2cell[NF0:NF1 - ny * nz, 3] = 2
        face2cell[NF1 - ny * nz:NF1, 1] = idx[-1].flatten()
        face2cell[NF1 - ny * nz:NF1, 3] = 3

        # y direction
        c = np.transpose(idx, (1, 2, 0))
        NF0 = NF1
        NF1 += nx * nz
        face2cell[NF0:NF1, 0] = c[0].flatten()
        face2cell[NF0:NF1, 1] = c[0].flatten()
        face2cell[NF0:NF1, 2:4] = 4

        NF0 = NF1
        NF1 += nx * ny * nz
        face2cell[NF0:NF1, 0] = c.flatten()
        face2cell[NF0:NF1, 2] = 5
        face2cell[NF0:NF1 - nx * nz, 1] = c[1:].flatten()
        face2cell[NF0:NF1 - nx * nz, 3] = 4
        face2cell[NF1 - nx * nz:NF1, 1] = c[-1].flatten()
        face2cell[NF1 - nx * nz:NF1, 3] = 5

        # z direction
        c = np.transpose(idx, (2, 0, 1))
        NF0 = NF1
        NF1 += nx * ny
        face2cell[NF0:NF1, 0] = c[0].flatten()
        face2cell[NF0:NF1, 1] = c[0].flatten()
        face2cell[NF0:NF1, 2:4] = 0

        NF0 = NF1
        NF1 += nx * ny * nz
        face2cell[NF0:NF1, 0] = c.flatten()
        face2cell[NF0:NF1, 2] = 1
        face2cell[NF0:NF1 - nx * ny, 1] = c[1:].flatten()
        face2cell[NF0:NF1 - nx * ny, 3] = 0
        face2cell[NF1 - nx * ny:NF1, 1] = c[-1].flatten()
        face2cell[NF1 - nx * ny:NF1, 3] = 1

        return face2cell

    @property
    def edge(self):
        NN = self.NN
        NE = self.NE

        nx = self.nx
        ny = self.ny
        nz = self.nz
        idx = np.arange(NN).reshape(nx + 1, ny + 1, nz + 1)
        edge = np.zeros((NE, 2), dtype=np.int)

        # NE0 = 0
        # NE1 = (ny + 1) * nz * (nx + 1)
        # J = np.ones(nz + 1, dtype=np.int)
        # J[1:-1] = 2
        # I = np.repeat(range(nz + 1), J)
        # edge[NE0:NE1, :] = idx[:, :, I].reshape(-1, 2)
        NE0 = 0
        NE1 = (nx + 1) * (ny + 1) * nz
        c = np.transpose(idx, (0, 1, 2))[:, :, :-1]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + 1
        edge[0:(ny + 1) * nz, :] = edge[0:(ny + 1) * nz, [0, 1]]

        # NE0 = NE1
        # NE1 += (nx + 1) * ny * (nz + 1)
        # J = np.ones(ny + 1, dtype=np.int)
        # J[1:-1] = 2
        # I = np.repeat(range(ny + 1), J)
        # edge[NE0:NE1, :] = idx.transpose(0, 2, 1)[:, :, I].reshape(-1, 2)
        NE0 = NE1
        NE1 += (nx + 1) * ny * (nz + 1)
        c = np.transpose(idx, (0, 1, 2))[:, :-1, :]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + nz + 1
        edge[(NE1 - (ny + 1) * nz):NE1, :] = edge[(NE1 - (ny + 1) * nz):NE1, [0, 1]]

        # NE0 = NE1
        # NE1 += nx * (ny + 1) * (nz + 1)
        # J = np.ones(nx + 1, dtype=np.int)
        # J[1:-1] = 2
        # I = np.repeat(range(nx + 1), J)
        # edge[NE0:NE1, :] = idx.transpose(1, 2, 0)[:, :, I].reshape(-1, 2)
        NE0 = NE1
        NE1 += (nx + 1) * (ny + 1) * nz
        c = np.transpose(idx, (0, 1, 2))[:-1, :, :]
        edge[NE0:NE1, 0] = c.flatten()
        edge[NE0:NE1, 1] = edge[NE0:NE1, 0] + (ny + 1) * (nz + 1)
        edge[NE0:NE0 + (ny + 1) * (nz + 1), :] = edge[NE0:NE0 + (ny + 1) * (nz + 1), [0, 1]]
        return edge

    @property
    def cell2edge(self):
        NN = self.NN
        NE = self.NE
        edge = self.edge
        idx = range(1, NE + 1)
        p2p = csr_matrix((idx, (edge[:, 0], edge[:, 1])), shape=(NN, NN),
                         dtype=np.int)
        totalEdge = self.total_edge()
        cell2edge = np.asarray(p2p[totalEdge[:, 0], totalEdge[:, 1]]).reshape(-1, 12)
        return cell2edge - 1

    def total_edge(self):
        NC = self.NC
        cell = self.cell
        localEdge = self.localEdge
        totalEdge = cell[:, localEdge].reshape(-1, localEdge.shape[1])
        return np.sort(totalEdge, axis=1)

    def cell_to_node(self):
        """
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = np.repeat(range(NC), V)
        val = np.ones(self.V * NC, dtype=np.bool)
        cell2node = csr_matrix((val, (I, cell.flatten())), shape=(NC, NN), dtype=np.bool)
        return cell2node

    def cell_to_edge(self, sparse=False):
        """ The neighbor information of cell to edge
        """
        if sparse == False:
            return self.cell2edge
        else:
            NC = self.NC
            NE = self.NE
            cell2edge = coo_matrix((NC, NE), dtype=np.bool)
            E = self.E
            I = np.repeat(range(NC), E)
            val = np.ones(E * NC, dtype=np.bool)
            cell2edge = csr_matrix((val, (I, self.cell2edge.flatten())), shape=(NC, NE), dtype=np.bool)
            return cell2edge

    def cell_to_edge_sign(self, cell):
        NC = self.NC
        E = self.E
        cell2edgeSign = np.zeros((NC, E), dtype=np.bool)
        localEdge = self.localEdge
        for i, (j, k) in zip(range(E), localEdge):
            cell2edgeSign[:, i] = cell[:, j] < cell[:, k]
        return cell2edgeSign

    def cell_to_face(self, sparse=False):
        NC = self.NC
        NF = self.NF
        face2cell = self.face2cell
        if sparse == False:
            F = self.F
            cell2face = np.zeros((NC, F), dtype=np.int)
            cell2face[face2cell[:, 0], face2cell[:, 2]] = range(NF)
            cell2face[face2cell[:, 1], face2cell[:, 3]] = range(NF)
            return cell2face
        else:
            val = np.ones((2 * NF,), dtype=np.bool)
            I = face2cell[:, [0, 1]].flatten()
            J = np.repeat(range(NF), 2)
            cell2face = csr_matrix((val, (I, J)), shape=(NC, NF), dtype=np.bool)
            return cell2face

    def cell_to_cell(self, return_sparse=False,
                     return_boundary=True, return_array=False):
        """ Get the adjacency information of cells
        """
        if return_array:
            return_sparse = False
            return_boundary = False

        NC = self.NC
        NF = self.NF
        face2cell = self.face2cell
        if (return_sparse == False) & (return_array == False):
            F = self.F
            cell2cell = np.zeros((NC, F), dtype=np.int)
            cell2cell[face2cell[:, 0], face2cell[:, 2]] = face2cell[:, 1]
            cell2cell[face2cell[:, 1], face2cell[:, 3]] = face2cell[:, 0]
            return cell2cell

        val = np.ones((NF,), dtype=np.bool)
        if return_boundary:
            cell2cell = coo_matrix(
                (val, (face2cell[:, 0], face2cell[:, 1])),
                shape=(NC, NC), dtype=np.bool)
            cell2cell += coo_matrix(
                (val, (face2cell[:, 1], face2cell[:, 0])),
                shape=(NC, NC), dtype=np.bool)
            return cell2cell.tocsr()
        else:
            isInFace = (face2cell[:, 0] != face2cell[:, 1])
            cell2cell = coo_matrix(
                (val[isInFace], (face2cell[isInFace, 0], face2cell[isInFace, 1])),
                shape=(NC, NC), dtype=np.bool)
            cell2cell += coo_matrix(
                (val[isInFace], (face2cell[isInFace, 1], face2cell[isInFace, 0])),
                shape=(NC, NC), dtype=np.bool)
            cell2cell = cell2cell.tocsr()
            if return_array == False:
                return cell2cell
            else:
                nn = cell2cell.sum(axis=1).reshape(-1)
                _, adj = cell2cell.nonzero()
                adjLocation = np.zeros(NC + 1, dtype=np.int32)
                adjLocation[1:] = np.cumsum(nn)
                return adj.astype(np.int32), adjLocation

    def face_to_node(self, return_sparse=False):

        face = self.face
        FE = self.localFace.shape[1]
        if return_sparse == False:
            return face
        else:
            N = self.N
            NF = self.NF
            I = np.repeat(range(NF), FE)
            val = np.ones(FE * NF, dtype=np.bool)
            face2node = csr_matrix((val, (I, face)), shape=(NF, N), dtype=np.bool)
            return face2node

    def face_to_edge(self, return_sparse=False):
        cell2edge = self.cell2edge
        face2cell = self.face2cell
        localFace2edge = self.localFace2edge
        FE = localFace2edge.shape[1]
        face2edge = cell2edge[face2cell[:, [0]], localFace2edge[face2cell[:, 2]]]
        if return_sparse == False:
            return face2edge
        else:
            NF = self.NF
            NE = self.NE
            I = np.repeat(range(NF), FE)
            J = face2edge.flatten()
            val = np.ones(FE * NF, dtype=np.bool)
            f2e = csr_matrix((val, (I, J)), shape=(NF, NE), dtype=np.bool)
            return f2e

    def face_to_face(self):
        face2edge = self.face_to_edge()
        return face2edge * face2edge.transpose()

    def face_to_cell(self, return_sparse=False):
        if return_sparse == False:
            return self.face2cell
        else:
            NC = self.NC
            NF = self.NF
            I = np.repeat(range(NF), 2)
            J = self.face2cell[:, [0, 1]].flatten()
            val = np.ones(2 * NF, dtype=np.bool)
            face2cell = csr_matrix((val, (I, J)), shape=(NF, NC), dtype=np.bool)
            return face2cell

    def edge_to_node(self, return_sparse=False):
        NN = self.NN
        NE = self.NE
        edge = self.edge
        if return_sparse == False:
            return edge
        else:
            edge = self.edge
            I = np.repeat(range(NE), 2)
            J = edge.flatten()
            val = np.ones(2 * NE, dtype=np.bool)
            edge2node = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool)
            return edge2node

    def edge_to_edge(self):
        edge2node = self.edge_to_node()
        return edge2node * edge2node.transpose()

    def edge_to_face(self):
        NF = self.NF
        NE = self.NE
        face2edge = self.face_to_edge()
        FE = face2edge.shape[1]
        I = face2edge.flatten()
        J = np.repeat(range(NF), FE)
        val = np.ones(FE * NF, dtype=np.bool)
        edge2face = csr_matrix((val, (I, J)), shape=(NE, NF), dtype=np.bool)
        return edge2face

    def edge_to_cell(self, localidx=False):
        NC = self.NC
        NE = self.NE
        cell2edge = self.cell2edge
        I = cell2edge.flatten()
        E = self.E
        J = np.repeat(range(NC), E)
        val = np.ones(E * NC, dtype=np.bool)
        edge2cell = csr_matrix((val, (I, J)), shape=(NE, NC), dtype=np.bool)
        return edge2cell

    def node_to_node(self):
        """ The neighbor information of nodes
        """
        NN = self.NN
        NE = self.NE
        edge = self.edge
        I = edge.flatten()
        J = edge[:, [1, 0]].flatten()
        val = np.ones((2 * NE,), dtype=np.bool)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN), dtype=np.bool)
        return node2node

    def node_to_edge(self):
        NN = self.NN
        NE = self.NE

        edge = self.edge
        I = edge.flatten()
        J = np.repeat(range(NE), 2)
        val = np.ones(2 * NE, dtype=np.bool)
        node2edge = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool)
        return node2edge

    def node_to_face(self):
        NN = self.NN
        NF = self.NF

        face = self.face
        FV = face.shape[1]

        I = face.flatten()
        J = np.repeat(range(NF), FV)
        val = np.ones(FV * NF, dtype=np.bool)
        node2face = csr_matrix((val, (I, J)), shape=(NF, NN), dtype=np.bool)
        return node2face

    def node_to_cell(self, return_local_index=False):
        """
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = cell.flatten()
        J = np.repeat(range(NC), V)

        if return_local_index == True:
            val = ranges(V * np.ones(NC, dtype=np.int), start=1)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.int)
        else:
            val = np.ones(V * NC, dtype=np.bool)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool)
        return node2cell

    def boundary_node_flag(self):
        NN = self.NN
        face = self.face
        isBdFace = self.boundary_face_flag()
        isBdPoint = np.zeros((NN,), dtype=np.bool)
        isBdPoint[face[isBdFace, :]] = True
        return isBdPoint

    def boundary_edge_flag(self):
        NE = self.NE
        face2edge = self.face_to_edge()
        isBdFace = self.boundary_face_flag()
        isBdEdge = np.zeros((NE,), dtype=np.bool)
        isBdEdge[face2edge[isBdFace, :]] = True
        return isBdEdge

    def boundary_face_flag(self):
        NF = self.NF
        face2cell = self.face_to_cell()
        return face2cell[:, 0] == face2cell[:, 1]

    def boundary_cell_flag(self):
        NC = self.NC
        face2cell = self.face_to_cell()
        isBdFace = self.boundary_face_flag()
        isBdCell = np.zeros((NC,), dtype=np.bool)
        isBdCell[face2cell[isBdFace, 0]] = True
        return isBdCell

    def boundary_node_index(self):
        isBdPoint = self.boundary_node_flag()
        idx, = np.nonzero(isBdPoint)
        return idx

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdPoint)
        return idx

    def boundary_face_index(self):
        isBdFace = self.boundary_face_flag()
        idx, = np.nonzero(isBdFace)
        return idx

    def z_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange((nx+1) * (ny+1) * nz)

    def y_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange((nx+1) * (ny+1) * nz, (nx+1) * (ny+1) * nz + (nx+1) * ny * (nz+1))

    def x_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        return np.arange((nx+1) * (ny+1) * nz + (nx+1) * ny * (nz+1), NE)

    def z_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        isZDEdge = np.zeros(NE, dtype=np.bool)
        isZDEdge[:(nx + 1) * (ny + 1) * nz] = True
        return isZDEdge

    def y_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        isYDEdge = np.zeros(NE, dtype=np.bool)
        isYDEdge[(nx + 1) * (ny + 1) * nz:-nx * (ny + 1) * (nz + 1)] = True
        return isYDEdge

    def x_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NE = self.NE
        isXDEdge = np.zeros(NE, dtype=np.bool)
        isXDEdge[-nx * (ny + 1) * (nz + 1):] = True
        return isXDEdge

    def z_direction_face_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange(nx * ny * (nz + 1))

    def y_direction_face_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        return np.arange(nx * ny * (nz + 1), nx * ny * (nz + 1) + nx * (ny + 1) * nz)

    def x_direction_face_index(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        return np.arange(nx * ny * (nz + 1) + nx * (ny + 1) * nz, NF)

    def z_direction_face_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        isXDFace = np.zeros(NF, dtype=np.bool)
        isXDFace[:nx * ny * (nz + 1)] = True
        return isXDFace

    def y_direction_face_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        isYDFace = np.zeros(NF, dtype=np.bool)
        isYDFace[nx * ny * (nz + 1):-(nx + 1) * ny * nz] = True
        return isYDFace

    def x_direction_face_flag(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        NF = self.NF
        isZDFace = np.zeros(NF, dtype=np.bool)
        isZDFace[-(nx + 1) * ny * nz:] = True
        return isZDFace

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx
