import numpy as np

from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve

from ..mesh import UniformMesh2d, UniformMesh2dFunction
from ..mesh import UniformMesh3d, UniformMesh3dFunction

class BackgroundMeshInterpolationAlg2D():
    def __init__(self, box = [0, 1, 0, 1], nx = 10, ny = 10):
        extent = np.array([0, nx+1, 0, ny+1], dtype=np.int_)
        h      = np.array([(box[1]-box[0])/nx, (box[3]-box[2])/ny], dtype=np.float_)
        origin = np.array([box[0], box[2]], dtype=np.float_)
        self.mesh = UniformMesh2d(extent, h, origin)

    def mass_matrix(self):
        h = self.mesh.h
        Mc = np.array([[4., 2., 2., 1.],
                       [2., 4., 1., 2.],
                       [2., 1., 4., 2.],
                       [1., 2., 2., 4.]], dtype=np.float_)*h[0]*h[1]/36
        cell2node = self.mesh.entity('cell')
        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells()

        data = np.broadcast_to(Mc, (NC, 4, 4))
        I = np.broadcast_to(cell2node[..., None], (NC, 4, 4))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 4, 4))
        M = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return M

    def stiff_matrix(self):
        h = self.mesh.h
        S0c = np.array([[ 2.,  1., -2., -1.],
                        [ 1.,  2., -1., -2.],
                        [-2., -1.,  2.,  1.],
                        [-1., -2.,  1.,  2.]], dtype=np.float_)*h[1]/h[0]/6
        S1c = np.array([[ 2., -2.,  1., -1.],
                        [-2.,  2., -1.,  1.],
                        [ 1., -1.,  2., -2.],
                        [-1.,  1., -2.,  2.]], dtype=np.float_)*h[0]/h[1]/6
        Sc = S0c + S1c
        cell2node = self.mesh.entity('cell')
        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells()

        data = np.broadcast_to(Sc, (NC, 4, 4))
        I = np.broadcast_to(cell2node[..., None], (NC, 4, 4))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 4, 4))
        S = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return S

    def nabla_2_matrix(self):
        h = self.mesh.h
        N2c = np.array([[ 1., -1., -1.,  1.],
                        [-1.,  1.,  1., -1.],
                        [-1.,  1.,  1., -1.],
                        [ 1., -1., -1.,  1.]], dtype=np.float_)*4/(h[1]*h[0])
        cell2node = self.mesh.entity('cell')
        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells()

        data = np.broadcast_to(N2c, (NC, 4, 4))
        I = np.broadcast_to(cell2node[..., None], (NC, 4, 4))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 4, 4))
        N2 = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return N2

    def nabla_jump_matrix(self):
        h, nx, ny = self.mesh.h, self.mesh.ds.nx, self.mesh.ds.ny
        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells()

        Jumpe = np.array([[ 2.,  1., -4., -2.,  2.,  1.],
                          [ 1.,  2., -2., -4.,  1.,  2.],
                          [-4., -2.,  8.,  4., -4., -2.],
                          [-2., -4.,  4.,  8., -2., -4.],
                          [ 2.,  1., -4., -2.,  2.,  1.],
                          [ 1.,  2., -2., -4.,  1.,  2.]])
        Jumpey = Jumpe*(h[1]/h[0]/h[0]/6)
        edge = self.mesh.entity('edge')
        edgex = edge[:nx*(ny+1)].reshape(nx, ny+1, 2)
        edgey = edge[nx*(ny+1):].reshape(nx+1, ny, 2)

        edgey2dof = np.zeros([nx-1, ny, 6], dtype=np.int_)
        edgey2dof[..., 0:2] = edgey[1:-1]-ny-1
        edgey2dof[..., 2:4] = edgey[1:-1]
        edgey2dof[..., 4:6] = edgey[1:-1]+ny+1

        data = np.broadcast_to(Jumpey, (nx-1, ny, 6, 6))
        I = np.broadcast_to(edgey2dof[..., None], data.shape)
        J = np.broadcast_to(edgey2dof[..., None, :], data.shape)
        Jump = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))

        Jumpex = Jumpe*(h[0]/h[1]/h[1]/6)
        edgex2dof = np.zeros([nx, ny-1, 6], dtype=np.int_)
        edgex2dof[..., 0:2] = edgex[:, 1:-1]-1
        edgex2dof[..., 2:4] = edgex[:, 1:-1]
        edgex2dof[..., 4:6] = edgex[:, 1:-1]+1

        data = np.broadcast_to(Jumpex, (nx, ny-1, 6, 6))
        I = np.broadcast_to(edgex2dof[..., None], data.shape)
        J = np.broadcast_to(edgex2dof[..., None, :], data.shape)
        Jump += csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return Jump

    def source_vector(self, f):
        cellarea = self.mesh.cell_area()
        cell2node = self.mesh.entity('cell')
        cellbar = self.mesh.entity_barycenter('cell')

        NN = self.mesh.number_of_nodes()
        NC = self.mesh.number_of_cells()

        node = self.mesh.entity('node')
        cell = self.mesh.entity('cell')

        #fval = f(node[cell])*cellarea/4 # (NC, )

        fval = f(cellbar).reshape(-1) # (NC, )
        fval = fval*cellarea/4
        fval = np.broadcast_to(fval[:, None], (NC, 4))

        F = np.zeros(NN, dtype=np.float_)
        np.add.at(F, cell2node, fval)
        return F

    def interpolation_with_sample_points(self, point, val, alpha=[10, 0.001, 0.01, 0.1]):
        '''!
        @brief 将 point, val 插值为网格函数
        @param point : 样本点
        @param val : 样本点的值
        '''
        mesh = self.mesh
        h, origin, nx, ny = mesh.h, mesh.origin, mesh.ds.nx, mesh.ds.ny
        cell = mesh.entity('cell').reshape(nx, ny, 4)

        NS = len(point) 
        NN = mesh.number_of_nodes()

        Xp = (point-origin)/h # (NS, 2)
        cellIdx = Xp.astype(np.int_) # 样本点所在单元
        xval = Xp - cellIdx 

        I = np.repeat(np.arange(NS), 4)
        J = cell[cellIdx[:, 0], cellIdx[:, 1]]
        data = np.zeros([NS, 4], dtype=np.float_)
        data[:, 0] = (1-xval[:, 0])*(1-xval[:, 1])
        data[:, 1] = (1-xval[:, 0])*xval[:, 1]
        data[:, 2] = xval[:, 0]*(1-xval[:, 1])
        data[:, 3] = xval[:, 0]*xval[:, 1]

        A = csr_matrix((data.flat, (I, J.flat)), (NS, NN), dtype=np.float_)
        B = self.stiff_matrix()
        C = self.nabla_2_matrix()
        D = self.nabla_jump_matrix()

        ### 标准化残量
        if 1:
            y = val-np.min(val)+0.01
            from scipy.sparse import spdiags
            Diag = spdiags(1/y, 0, NS, NS)
            A = Diag@A

            S = alpha[0]*A.T@A + alpha[1]*B + alpha[2]*C + alpha[3]*D
            F = alpha[0]*A.T@np.ones(NS, dtype=np.float_)
            f = spsolve(S, F).reshape(nx+1, ny+1)+np.min(val)-0.01
        else:
            S = alpha[0]*A.T@A + alpha[1]*B + alpha[2]*C + alpha[3]*D
            F = alpha[0]*A.T@val
            f = spsolve(S, F).reshape(nx+1, ny+1)
        return UniformMesh2dFunction(mesh, f)

class BackgroundMeshInterpolationAlg3D():
    def __init__(self, box = [0, 1, 0, 1, 0, 1], nx = 10, ny = 10, nz = 10):
        h = np.array([(box[1]-box[0])/nx, (box[3]-box[2])/ny, (box[5]-box[4])/nz], dtype=np.float_)
        extent = np.array([0, nx+1, 0, ny+1, 0, nz+1], dtype=np.int_)
        origin = np.array([box[0], box[2], box[4]], dtype=np.float_)
        self.mesh = UniformMesh3d(extent, h, origin)

    def mass_matrix(self):
        mesh = self.mesh
        h = mesh.h
        Mc = np.array([[8., 4., 2., 4., 4., 2., 1., 2.],
                       [4., 8., 4., 2., 2., 4., 2., 1.],
                       [2., 4., 8., 4., 1., 2., 4., 2.],
                       [4., 2., 4., 8., 2., 1., 2., 4.],
                       [4., 2., 1., 2., 8., 4., 2., 4.],
                       [2., 4., 2., 1., 4., 8., 4., 2.],
                       [1., 2., 4., 2., 2., 4., 8., 4.],
                       [2., 1., 2., 4., 4., 2., 4., 8.]])*(h[0]*h[1]*h[2]/36/6)
        cell2node = mesh.entity('cell')
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        data = np.broadcast_to(Mc, (NC, 8, 8))
        I = np.broadcast_to(cell2node[..., None], (NC, 8, 8))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 8, 8))
        M = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return M

    def stiff_matrix(self):
        mesh = self.mesh
        h = mesh.h
        S0c = np.array([[ 4.,  2.,  1.,  2., -4., -2., -1., -2.],
                        [ 2.,  4.,  2.,  1., -2., -4., -2., -1.],
                        [ 1.,  2.,  4.,  2., -1., -2., -4., -2.],
                        [ 2.,  1.,  2.,  4., -2., -1., -2., -4.],
                        [-4., -2., -1., -2.,  4.,  2.,  1.,  2.],
                        [-2., -4., -2., -1.,  2.,  4.,  2.,  1.],
                        [-1., -2., -4., -2.,  1.,  2.,  4.,  2.],
                        [-2., -1., -2., -4.,  2.,  1.,  2.,  4.]])
        S1c = np.array([[ 4., -4., -2.,  2.,  2., -2., -1.,  1.],
                        [-4.,  4.,  2., -2., -2.,  2.,  1., -1.],
                        [-2.,  2.,  4., -4., -1.,  1.,  2., -2.],
                        [ 2., -2., -4.,  4.,  1., -1., -2.,  2.],
                        [ 2., -2., -1.,  1.,  4., -4., -2.,  2.],
                        [-2.,  2.,  1., -1., -4.,  4.,  2., -2.],
                        [-1.,  1.,  2., -2., -2.,  2.,  4., -4.],
                        [ 1., -1., -2.,  2.,  2., -2., -4.,  4.]])
        S2c = np.array([[ 4.,  2., -2., -4.,  2.,  1., -1., -2.],
                        [ 2.,  4., -4., -2.,  1.,  2., -2., -1.],
                        [-2., -4.,  4.,  2., -1., -2.,  2.,  1.],
                        [-4., -2.,  2.,  4., -2., -1.,  1.,  2.],
                        [ 2.,  1., -1., -2.,  4.,  2., -2., -4.],
                        [ 1.,  2., -2., -1.,  2.,  4., -4., -2.],
                        [-1., -2.,  2.,  1., -2., -4.,  4.,  2.],
                        [-2., -1.,  1.,  2., -4., -2.,  2.,  4.]])
        S0c *= h[1]*h[2]/h[0]/36
        S1c *= h[0]*h[2]/h[1]/36
        S2c *= h[0]*h[1]/h[2]/36

        Sc = S0c + S1c + S2c
        cell2node = mesh.entity('cell')
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        data = np.broadcast_to(Sc, (NC, 8, 8))
        I = np.broadcast_to(cell2node[..., None], (NC, 8, 8))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 8, 8))
        S = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return S

    def grad_2_matrix(self):
        mesh = self.mesh
        h = mesh.h
        N201 = np.array([[ 2., -2., -1.,  1., -2.,  2.,  1., -1.],
                         [-2.,  2.,  1., -1.,  2., -2., -1.,  1.],
                         [-1.,  1.,  2., -2.,  1., -1., -2.,  2.],
                         [ 1., -1., -2.,  2., -1.,  1.,  2., -2.],
                         [-2.,  2.,  1., -1.,  2., -2., -1.,  1.],
                         [ 2., -2., -1.,  1., -2.,  2.,  1., -1.],
                         [ 1., -1., -2.,  2., -1.,  1.,  2., -2.],
                         [-1.,  1.,  2., -2.,  1., -1., -2.,  2.]])*2
        N202 = np.array([[ 2.,  1., -1., -2., -2., -1.,  1.,  2.],
                         [ 1.,  2., -2., -1., -1., -2.,  2.,  1.],
                         [-1., -2.,  2.,  1.,  1.,  2., -2., -1.],
                         [-2., -1.,  1.,  2.,  2.,  1., -1., -2.],
                         [-2., -1.,  1.,  2.,  2.,  1., -1., -2.],
                         [-1., -2.,  2.,  1.,  1.,  2., -2., -1.],
                         [ 1.,  2., -2., -1., -1., -2.,  2.,  1.],
                         [ 2.,  1., -1., -2., -2., -1.,  1.,  2.]])*2
        N212 = np.array([[ 2., -2.,  2., -2.,  1., -1.,  1., -1.],
                         [-2.,  2., -2.,  2., -1.,  1., -1.,  1.],
                         [ 2., -2.,  2., -2.,  1., -1.,  1., -1.],
                         [-2.,  2., -2.,  2., -1.,  1., -1.,  1.],
                         [ 1., -1.,  1., -1.,  2., -2.,  2., -2.],
                         [-1.,  1., -1.,  1., -2.,  2., -2.,  2.],
                         [ 1., -1.,  1., -1.,  2., -2.,  2., -2.],
                         [-1.,  1., -1.,  1., -2.,  2., -2.,  2.]])*2
        N201 *= h[2]/h[0]/h[1]/6
        N202 *= h[1]/h[0]/h[2]/6
        N212 *= h[0]/h[1]/h[2]/6
        N2c = N201 + N202 + N212
        cell2node = mesh.entity('cell')
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        data = np.broadcast_to(N2c, (NC, 8, 8))
        I = np.broadcast_to(cell2node[..., None], (NC, 8, 8))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 8, 8))
        N2 = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return N2

    def grad_jump_matrix(self):
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        h, nx, ny, nz = mesh.h, mesh.ds.nx, mesh.ds.ny, mesh.ds.nz

        face = mesh.entity('face')
        facex = face[:(nx + 1)*ny*nz].reshape(nx+1, ny, nz, 4)
        facey = face[(nx + 1)*ny*nz:(nx+1)*ny*nz+nx*(ny+1)*nz].reshape(nx, ny+1,
                nz, 4)
        facez = face[(nx+1)*ny*nz+nx*(ny+1)*nz:].reshape(nx, ny, nz+1, 4)

        Jumpf = np.array([[ 4.,  2.,  2.,  1., -8., -4., -4., -2.,  4.,  2.,  2.,  1.],
                          [ 2.,  4.,  1.,  2., -4., -8., -2., -4.,  2.,  4.,  1.,  2.],
                          [ 2.,  1.,  4.,  2., -4., -2., -8., -4.,  2.,  1.,  4.,  2.],
                          [ 1.,  2.,  2.,  4., -2., -4., -4., -8.,  1.,  2.,  2.,  4.],
                          [-8., -4., -4., -2., 16.,  8.,  8.,  4., -8., -4., -4., -2.],
                          [-4., -8., -2., -4.,  8., 16.,  4.,  8., -4., -8., -2., -4.],
                          [-4., -2., -8., -4.,  8.,  4., 16.,  8., -4., -2., -8., -4.],
                          [-2., -4., -4., -8.,  4.,  8.,  8., 16., -2., -4., -4., -8.],
                          [ 4.,  2.,  2.,  1., -8., -4., -4., -2.,  4.,  2.,  2.,  1.],
                          [ 2.,  4.,  1.,  2., -4., -8., -2., -4.,  2.,  4.,  1.,  2.],
                          [ 2.,  1.,  4.,  2., -4., -2., -8., -4.,  2.,  1.,  4.,  2.],
                          [ 1.,  2.,  2.,  4., -2., -4., -4., -8.,  1.,  2.,  2.,  4.]])
        Jumpfx = Jumpf * (h[1]*h[2]/h[0]**2/36) 

        facex2dof = np.zeros([nx-1, ny, nz, 12], dtype=np.int_)
        facex2dof[..., 0: 4] = facex[1:-1]-(ny+1)*(nz+1)
        facex2dof[..., 4: 8] = facex[1:-1]
        facex2dof[..., 8:12] = facex[1:-1]+(ny+1)*(nz+1)

        data = np.broadcast_to(Jumpfx, (nx-1, ny, nz, 12, 12))
        I = np.broadcast_to(facex2dof[..., None], data.shape)
        J = np.broadcast_to(facex2dof[..., None, :], data.shape)
        Jump = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))

        Jumpfy = Jumpf * (h[0]*h[2]/h[1]**2/36) 
        facey2dof = np.zeros([nx, ny-1, nz, 12], dtype=np.int_)
        facey2dof[..., 4: 8] = facey[:, 1:-1, :, [0, 2, 1, 3]]
        facey2dof[..., 0: 4] = facey2dof[..., 4:8]-(nz+1)
        facey2dof[..., 8:12] = facey2dof[..., 4:8]+(nz+1)

        data = np.broadcast_to(Jumpfy, (nx, ny-1, nz, 12, 12))
        I = np.broadcast_to(facey2dof[..., None], data.shape)
        J = np.broadcast_to(facey2dof[..., None, :], data.shape)
        Jump += csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))

        Jumpfz = Jumpf * (h[0]*h[1]/h[2]**2/36) 
        facez2dof = np.zeros([nx, ny, nz-1, 12], dtype=np.int_)
        facez2dof[..., 4: 8] = facez[:, :, 1:-1]
        facez2dof[..., 0: 4] = facez2dof[..., 4:8]-1
        facez2dof[..., 8:12] = facez2dof[..., 4:8]+1

        data = np.broadcast_to(Jumpfz, (nx, ny, nz-1, 12, 12))
        I = np.broadcast_to(facez2dof[..., None], data.shape)
        J = np.broadcast_to(facez2dof[..., None, :], data.shape)
        Jump += csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return Jump

    def source_vector(self, f):
        mesh = self.mesh
        cellvol = mesh.cell_volume()
        cell2node = mesh.entity('cell')
        cellbar = mesh.entity_barycenter('cell')

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        node = mesh.entity('node')
        cell = mesh.entity('cell')

        #fval = f(node[cell])*cellarea/4 # (NC, )

        fval = f(cellbar).reshape(-1) # (NC, )
        fval = fval*cellvol/8
        fval = np.broadcast_to(fval[:, None], (NC, 8))

        F = np.zeros(NN, dtype=np.float_)
        np.add.at(F, cell2node, fval)
        return F


    def interpolation_with_sample_points(self, point, val, alpha=[10, 0.001, 0.01, 0.1]):
        '''!
        @brief 将 point, val 插值为网格函数
        @param point : 样本点
        @param val : 样本点的值
        '''
        mesh = self.mesh
        h, origin, nx, ny, nz = mesh.h, mesh.origin, mesh.ds.nx, mesh.ds.ny, mesh.ds.nz
        cell = mesh.entity('cell').reshape(nx, ny, nz, 8)

        NS = len(point) 
        NN = mesh.number_of_nodes()

        Xp = (point-origin)/h # (NS, 3)
        cellIdx = Xp.astype(np.int_) # 样本点所在单元
        xval = Xp - cellIdx 

        I = np.repeat(np.arange(NS), 8)
        J = cell[cellIdx[:, 0], cellIdx[:, 1], cellIdx[:, 2]]
        data = np.zeros([NS, 8], dtype=np.float_)
        data[:, 0] = (1-xval[:, 0])*(1-xval[:, 1])*(1-xval[:, 2])
        data[:, 1] = (1-xval[:, 0])*(1-xval[:, 1])*xval[:, 2]
        data[:, 2] = (1-xval[:, 0])*xval[:, 1]*(1-xval[:, 2])
        data[:, 3] = (1-xval[:, 0])*xval[:, 1]*xval[:, 2]
        data[:, 4] = xval[:, 0]*(1-xval[:, 1])*(1-xval[:, 2])
        data[:, 5] = xval[:, 0]*(1-xval[:, 1])*xval[:, 2]
        data[:, 6] = xval[:, 0]*xval[:, 1]*(1-xval[:, 2])
        data[:, 7] = xval[:, 0]*xval[:, 1]*xval[:, 2]

        A = csr_matrix((data.flat, (I, J.flat)), (NS, NN), dtype=np.float_)
        B = self.stiff_matrix()
        C = self.grad_2_matrix()
        D = self.grad_jump_matrix()

        ### 标准化残量
        if 0:
            y = val-np.min(val)+0.01
            from scipy.sparse import spdiags
            Diag = spdiags(1/y, 0, NS, NS)
            A = Diag@A

            S = alpha[0]*A.T@A + alpha[1]*B + alpha[2]*C + alpha[3]*D
            F = alpha[0]*A.T@np.ones(NS, dtype=np.float_)
            f = spsolve(S, F).reshape(nx+1, ny+1)+np.min(y)-0.01
        else:
            S = alpha[0]*A.T@A + alpha[1]*B + alpha[2]*C + alpha[3]*D
            F = alpha[0]*A.T@val
            f = spsolve(S, F).reshape(nx+1, ny+1, nz+1)
        return UniformMesh3dFunction(mesh, f)
