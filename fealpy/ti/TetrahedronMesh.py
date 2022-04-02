import taichi as ti
import numpy as np
from scipy.sparse import csr_matrix

class TetrahedronMeshDataStructure():
    localFace = ti.Matrix([(1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)])
    localEdge = ti.Matrix([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    localFace2edge = ti.Matrix([(5, 4, 3), (5, 1, 2), (4, 2, 0), (3, 0, 1)])

def construct(cell):
    NC = cell.shape[0]

    localFace = np.array([(1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)])
    localEdge = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    #构建 face
    totalFace = cell[:, localFace].reshape(NC*4, 3)

    _, i0, j = np.unique(
            np.sort(totalFace, axis=1),
            return_index=True,
            return_inverse=True,
            axis=0)
    face = totalFace[i0]

    # 构建 face2cell
    NF = i0.shape[0]
    face2cell = np.zeros((NF, 4), dtype=np.int_)

    i1 = np.zeros(NF, dtype=np.int_)
    i1[j] = np.arange(NC*4)

    face2cell[:, 0] = i0 // 4
    face2cell[:, 1] = i1 // 4
    face2cell[:, 2] = i0 % 4
    face2cell[:, 3] = i1 % 4
    cell2face = np.reshape(j, (NC, 4))

    totalEdge = cell[:, localEdge].reshape(-1, 2)
    edge, i2, j = np.unique(
            np.sort(totalEdge, axis=1),
            return_index=True,
            return_inverse=True,
            axis=0)
    cell2edge = np.reshape(j, (NC, 6))
    return face, edge, cell2edge, cell2face, face2cell

@ti.data_oriented
class TetrahedronMesh():
    def __init__(self, node, cell, itype=ti.u32, ftype=ti.f64):

        assert cell.shape[-1] == 4
        assert node.shape[-1] == 3

        self.itype = itype
        self.ftype = ftype

        NN = node.shape[0]
        self.node = ti.field(self.ftype, (NN, 3))
        self.node.from_numpy(node)

        NC = cell.shape[0]
        self.cell = ti.field(self.ftype, shape=(NC, 4))
        self.cell.from_numpy(cell)

        self.construct_data_structure(cell)


    def construct_data_structure(self, cell):
        """! 构造四面体网格的辅助数据结构
        """

        face, edge, cell2edge, cell2face, face2cell = construct(cell)

        NE = edge.shape[0]
        NF = face.shape[0]
        NC = cell.shape[0]

        self.edge = ti.field(self.itype, shape=(NE, 2))
        self.edge.from_numpy(edge)

        self.face = ti.field(self.itype, shape=(NF, 3))
        self.face.from_numpy(face)

        self.face2cell = ti.field(self.itype, shape=(NF, 4))
        self.face2cell.from_numpy(face2cell)

        self.cell2edge = ti.field(self.itype, shape=(NC, 6))
        self.cell2edge.from_numpy(cell2edge)

        self.cell2face = ti.field(self.itype, shape=(NC, 4))
        self.cell2face.from_numpy(cell2face)

    def multi_index_matrix(self, p):
        ldof = (p+1)*(p+2)*(p+3)//6
        idx = np.arange(1, ldof)
        idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
        idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
        idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
        idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
        multiIndex = np.zeros((ldof, 4), dtype=np.int_)
        multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
        multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
        multiIndex[1:, 1] = idx0 - idx2
        multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
        return multiIndex

    def number_of_local_interpolation_points(self, p):
        return (p+1)*(p+2)*(p+3)//6

    def number_of_global_interpolation_points(self, p):
        NP = self.number_of_nodes()
        if p > 1:
            NE = self.number_of_edges()
            NP += NE*(p-1)

        if p > 2:
            NF = self.number_of_faces()
            NP += NF*(p-2)*(p-1)//2

        if p > 3:
            NC = self.number_of_cells()
            NP += NC*(p-3)*(p-2)*(p-1)//6 
        return NP

    @ti.kernel 
    def interpolation_points(self, p: ti.u32, ipoints: ti.template()):
        NN = self.node.shape[0]
        NE = self.edge.shape[0]
        NF = self.face.shape[0]
        NC = self.cell.shape[0]
        GD = self.node.shape[1]

        for I in range(NN):
            print(I, ":")
            for d in range(GD):
                ipoints[I, d] = self.node[I, d]

        if p > 1:
            for e in range(NE):
                s1 = NN + e*(p-1)
                for i1 in range(1, p):
                    i0 = p - i1 # (i0, i1)
                    I = s1 + i1 - 1
                    print(I, ":")
                    for d in range(GD):
                        ipoints[I, d] = (
                                i0*self.node[self.edge[e, 0], d] + 
                                i1*self.node[self.edge[e, 1], d])/p
        if p > 2:
            fdof = (p-2)*(p-1)//2
            s0 = NN + (p-1)*NE
            for f in range(NF):
                s1 = s0 + f*fdof
                for line in range(0, p-2):
                    i0 = p - 2 - line 
                    for i2 in range(1, line+2):
                        i1 = p - i0 - i2 #(i0, i1, i2)
                        j = i1 + i2 - 2
                        I = s1 + j*(j+1)//2 + i2 - 1  
                        print(I, ":")
                        for d in range(GD):
                            ipoints[I, d] = (
                                    i0*self.node[self.face[f, 0], d] + 
                                    i1*self.node[self.face[f, 1], d] + 
                                    i2*self.node[self.face[f, 2], d])/p
        if p > 3:
            cdof = (p-3)*(p-2)*(p-1)//6
            s0 = NN + NE*(p-1) + NF*(p-2)*(p-1)//2
            for c in range(NC):
                s1 = s0 + c*cdof
                for level in range(0, p-3):
                    i0 = p - 3 - level
                    for line in range(1, level+2):
                        i1 = p - 2 - line
                        for i3 in range(1, line+2):
                            i2 = p - i0 - i1 - i3 #(i0, i1, i2, i3)
                            j0 = i1 + i2 + i3 - 3
                            j1 = i2 + i3 - 2
                            I = s1 + j0*(j0+1)*(j0+2)//6 + j1*(j1+1)//2 + i3 - 1
                            print(I, ":")
                            for d in range(GD):
                                ipoints[I, d] = (
                                        i0*self.node[self.cell[c, 0], d] + 
                                        i1*self.node[self.cell[c, 1], d] + 
                                        i2*self.node[self.cell[c, 2], d] + 
                                        i3*self.node[self.cell[c, 3], d])/p


    def geo_dimension(self):
        return 3

    def top_dimension(self):
        return 3

    def number_of_nodes(self):
        return self.node.shape[0]

    def number_of_edges(self):
        return self.edge.shape[0]

    def number_of_faces(self):
        return self.face.shape[0]

    def number_of_cells(self):
        return self.cell.shape[0]

    def entity(self, etype=2):
        if etype in {'cell', 3}:
            return self.cell
        elif etype in {'face', 2}:
            return self.face
        elif etype in {'edge', 1}:
            return self.edge
        elif etype in {'node', 0}:
            return self.node
        else:
            raise ValueError("`etype` is wrong!")

    def vtk_cell_type(self):
        VTK_TETRA = 10
        return VTK_TETRA

    def to_vtk(self, fname, nodedata=None, celldata=None):
        """

        Parameters
        ----------
        points: vtkPoints object
        cells:  vtkCells object
        pdata:  
        cdata:

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        from fealpy.mesh.vtk_extent import vtk_cell_index, write_to_vtu

        node = self.node.to_numpy()
        GD = self.geo_dimension()

        cell = self.cell.to_numpy(dtype=np.int_)
        cellType = self.vtk_cell_type()
        NV = cell.shape[-1]

        NC = self.number_of_cells()
        cell = np.r_['1', np.zeros((NC, 1), dtype=cell.dtype), cell]
        cell[:, 0] = NV

        print("Writting to vtk...")
        write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                nodedata=nodedata,
                celldata=celldata)

    @ti.func
    def cell_measure(self, c: ti.u32) -> ti.f64:
        """
        计算第i 个单元的体积测度
        """
        V = ti.Matrix.zero(ti.f64, 3, 3)
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                V[i, j] = self.node[self.cell[c, i+1], j] - self.node[self.cell[c, 0], j]
        vol = V.determinant()/6.0
        return vol 

    @ti.func
    def grad_lambda(self, c: ti.u32) -> (ti.types.matrix(4, 3, ti.f64), ti.f64):
        """
        计算第 c 个单元上重心坐标函数的梯度，以及单元的面积
        """
        vol = self.cell_measure(c)*6
        glambda = ti.Matrix.zero(ti.f64, 4, 3)
        v = ti.Matrix.zero(ti.f64, 2, 3)
        for i in ti.static(range(4)):
            j = self.ds.localFace[i, 0]
            k = self.ds.localFace[i, 1]
            m = self.ds.localFace[i, 2]
            for l in ti.static(range(3)):
                v[0, l] = self.node[self.cell[c, k], l] - self.node[self.cell[c, j], l]
                v[1, l] = self.node[self.cell[c, m], l] - self.node[self.cell[c, j], l]
            glambda[i, 0] = (v[0, 2]*v[1, 1] - v[0, 1]*v[1, 2])/vol
            glambda[i, 1] = (v[0, 0]*v[1, 2] - v[0, 2]*v[1, 0])/vol 
            glambda[i, 2] = (v[0, 1]*v[1, 0] - v[0, 0]*v[1, 1])/vol 
        return glambda, vol/6

    @ti.kernel
    def cell_stiff_matrices(self, S: ti.template()):
        """
        计算网格上的所有单元刚度矩阵
        """
        for c in range(self.cell.shape[0]):
            gphi, vol = self.grad_lambda(c) 

            S[c, 0, 0] = vol*(gphi[0, 0]*gphi[0, 0] + gphi[0, 1]*gphi[0, 1]+ gphi[0, 2]*gphi[0, 2])
            S[c, 0, 1] = vol*(gphi[0, 0]*gphi[1, 0] + gphi[0, 1]*gphi[1, 1]+ gphi[0, 2]*gphi[1, 2])
            S[c, 0, 2] = vol*(gphi[0, 0]*gphi[2, 0] + gphi[0, 1]*gphi[2, 1]+ gphi[0, 2]*gphi[2, 2])
            S[c, 0, 3] = vol*(gphi[0, 0]*gphi[3, 0] + gphi[0, 1]*gphi[3, 1]+ gphi[0, 2]*gphi[3, 2])

            S[c, 1, 0] = S[c, 0, 1]
            S[c, 1, 1] = vol*(gphi[1, 0]*gphi[1, 0] + gphi[1, 1]*gphi[1, 1]+ gphi[1, 2]*gphi[1, 2])
            S[c, 1, 2] = vol*(gphi[1, 0]*gphi[2, 0] + gphi[1, 1]*gphi[2, 1]+ gphi[1, 2]*gphi[2, 2])
            S[c, 1, 3] = vol*(gphi[1, 0]*gphi[3, 0] + gphi[1, 1]*gphi[3, 1]+ gphi[1, 2]*gphi[3, 2])

            S[c, 2, 0] = S[c, 0, 2]
            S[c, 2, 1] = S[c, 1, 2]
            S[c, 2, 2] = vol*(gphi[2, 0]*gphi[2, 0] + gphi[2, 1]*gphi[2, 1]+ gphi[2, 2]*gphi[2, 2])
            S[c, 2, 3] = vol*(gphi[2, 0]*gphi[3, 0] + gphi[2, 1]*gphi[3, 1]+ gphi[2, 2]*gphi[3, 2])

            S[c, 3, 0] = S[c, 0, 3]
            S[c, 3, 1] = S[c, 1, 3]
            S[c, 3, 2] = S[c, 2, 3]
            S[c, 3, 3] = l*(gphi[3, 0]*gphi[3, 0] + gphi[3, 1]*gphi[3, 1]+ gphi[3, 2]*gphi[3, 2])

    @ti.kernel
    def cell_mass_matrices(self, S: ti.template()):
        """
        计算网格上的所有单元质量矩阵
        """
        for c in range(self.cell.shape[0]):

            vol = self.cell_measure(c)
            c0 = vol/10.0
            c1 = vol/20.0

            S[c, 0, 0] = c0 
            S[c, 0, 1] = c1
            S[c, 0, 2] = c1
            S[c, 0, 3] = c1

            S[c, 1, 0] = c1 
            S[c, 1, 1] = c0  
            S[c, 1, 2] = c1
            S[c, 1, 3] = c1

            S[c, 2, 0] = c1 
            S[c, 2, 1] = c1 
            S[c, 2, 2] = c0 
            S[c, 2, 3] = c1 

            S[c, 3, 0] = c1 
            S[c, 3, 1] = c1 
            S[c, 3, 2] = c1 
            S[c, 3, 3] = c0 

    @ti.kernel
    def cell_convection_matrices(self, u: ti.template(), S:ti.template()):
        """! 计算网格上所有单元的对流矩阵

        The continuous weak foumulation

        \f[ (\\boldsymbol\cdot \\nabla\phi, v) \f]

        where \f$\\boldsymbol\f$ is the velocity field, \f$\phi\f$ is trial
        function, and \f$ v \f$ is test function.

        @param[in] u  Taichi field with shape (NN, 3)
        @param[in, out] S  Taichi field with shape (NN, 4, 4)

        See Also
        -------
        """
        for c in range(self.cell.shape[0]):
            gphi, vol = self.grad_lambda(c) 

            c0 = vol/10.0
            c1 = vol/20.0

            U = ti.Matrix.zero(ti.f64, 4, 3)

            for i in ti.static(range(3)):
                U[0, i] += u[self.cell[c, 0], i]*c0 
                U[0, i] += u[self.cell[c, 1], i]*c1 
                U[0, i] += u[self.cell[c, 2], i]*c1 
                U[0, i] += u[self.cell[c, 3], i]*c1

            for i in ti.static(range(3)):
                U[1, i] += u[self.cell[c, 0], i]*c1 
                U[1, i] += u[self.cell[c, 1], i]*c0 
                U[1, i] += u[self.cell[c, 2], i]*c1 
                U[1, i] += u[self.cell[c, 3], i]*c1

            for i in ti.static(range(3)):
                U[2, i] += u[self.cell[c, 0], i]*c1 
                U[2, i] += u[self.cell[c, 1], i]*c1 
                U[2, i] += u[self.cell[c, 2], i]*c0 
                U[2, i] += u[self.cell[c, 3], i]*c1

            for i in ti.static(range(3)):
                U[3, i] += u[self.cell[c, 0], i]*c1 
                U[3, i] += u[self.cell[c, 1], i]*c1 
                U[3, i] += u[self.cell[c, 2], i]*c1 
                U[3, i] += u[self.cell[c, 3], i]*c0

            for i in ti.static(range(4)):
                for j in ti.static(range(4)):
                    S[c, i, j] = U[i, 0]*gphi[j, 0] + U[i, 1]*gphi[j, 1] + U[i, 2]*gphi[j, 2]

    @ti.kernel
    def cell_source_vectors(self, f:ti.template(), bc:ti.template(), ws:ti.template(), F:ti.template()):
        """
        计算所有单元载荷
        """
        for c in range(self.cell.shape[0]):
            x0 = self.node[self.cell[c, 0], 0]
            y0 = self.node[self.cell[c, 0], 1]
            z0 = self.node[self.cell[c, 0], 2]

            x1 = self.node[self.cell[c, 1], 0]
            y1 = self.node[self.cell[c, 1], 1]
            z1 = self.node[self.cell[c, 1], 2]

            x2 = self.node[self.cell[c, 2], 0]
            y2 = self.node[self.cell[c, 2], 1]
            z2 = self.node[self.cell[c, 2], 2]

            x3 = self.node[self.cell[c, 3], 0]
            y3 = self.node[self.cell[c, 3], 1]
            z3 = self.node[self.cell[c, 3], 2]

            l1 = (x1 - x0)*((y2 - y0)*(z3 - z0) - (z2 - z0)*(y3 - y0))
            l2 = (y1 - y0)*((x3 - x0)*(z2 - z0) - (z3 - z0)*(x2 - x0))
            l3 = (z1 - x0)*((x2 - x0)*(y3 - y0) - (x3 - x0)*(y2 - y0))

            l = (l1 + l2 + l3)/6
            for q in ti.static(range(bc.n)):
                x = x0*bc[q, 0] + x1*bc[q, 1] + x2*bc[q, 2] + x3*bc[q, 3]
                y = y0*bc[q, 0] + y1*bc[q, 1] + y2*bc[q, 2] + y3*bc[q, 3]
                z = z0*bc[q, 0] + z1*bc[q, 1] + z2*bc[q, 2] + z3*bc[q, 3]
                r = f(x, y, z)
                for i in ti.static(range(4)):
                    F[c, i] += ws[q]*bc[q, i]*r

            for i in range(4):
                F[c, i] *= l

    def stiff_matrix(self, c=None):
        """
        组装总体刚度矩阵
        """
        NC = self.number_of_cells()

        K = ti.field(ti.f64, (NC, 4, 4))
        self.cell_stiff_matrices(K)

        M = K.to_numpy()
        if c is not None:
            M *= c # 目前假设 c 为一常数

        cell = self.cell.to_numpy()
        I = np.broadcast_to(cell[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell[:, None, :], shape=M.shape)

        NN = self.number_of_nodes()
        M = csr_matrix((K.to_numpy().flat, (I.flat, J.flat)), shape=(NN, NN))
        return M

    def mass_matrix(self, c=None):
        """
        组装总体质量矩阵
        """
        NC = self.number_of_cells()

        K = ti.field(ti.f64, (NC, 4, 4))
        self.cell_mass_matrices(K)

        M = K.to_numpy()
        if c is not None:
            M *= c

        cell = self.cell.to_numpy()
        I = np.broadcast_to(cell[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell[:, None, :], shape=M.shape)

        NN = self.number_of_nodes() 
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(NN, NN))
        return M

    def convection_matrix(self, u):
        """
        组装总体对流矩阵
        """

        NC = self.number_of_cells() 

        C = ti.field(ti.f64, (NC, 4, 4))
        self.cell_convection_matrices(u, C)

        M = C.to_numpy()

        cell = self.cell.to_numpy()
        I = np.broadcast_to(cell[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell[:, None, :], shape=M.shape)

        NN = self.number_of_nodes()
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(NN, NN))
        return M

    def source_vector(self, f):
        """
        组装总体载荷向量
        """
        NN = self.node.shape[0]
        NC = self.cell.shape[0]
        bc = ti.Matrix([
            [0.5854101966249680,	0.1381966011250110,
                0.1381966011250110,	0.1381966011250110],
            [0.1381966011250110,	0.5854101966249680,
                0.1381966011250110,	0.1381966011250110],
            [0.1381966011250110,	0.1381966011250110,
                0.5854101966249680,	0.1381966011250110],
            [0.1381966011250110,	0.1381966011250110,
                0.1381966011250110,	0.5854101966249680]], dt=ti.f64)
        ws = ti.Vector([0.25, 0.25, 0.25, 0.25], dt=ti.f64)

        F = ti.field(ti.f64, (NC, 4))
        self.cell_source_vectors(f, bc, ws, F)

        bb = F.to_numpy()
        F = np.zeros(NN, dtype=np.float64)
        cell = self.cell.to_numpy()
        np.add.at(F, cell, bb)
        return F

    @ti.kernel
    def linear_scalar_interpolation(self, f: ti.template(), R: ti.template()):
        """! 定义在三维区域中的标量函数的线性插值
        """
        for i in range(self.node.shape[0]):
            R[i] = f(self.node[i, 0], self.node[i, 1], self.node[i, 2])

    @ti.kernel
    def linear_vector_interpolation(self, f: ti.template(), R: ti.template()):
        """! 定义在三维区域中的向量函数的线性插值
        """
        for i in range(self.node.shape[0]):
            R[i, 0], R[i, 1], R[i, 2] = f(self.node[i, 0], self.node[i, 1], self.node[i, 2])
