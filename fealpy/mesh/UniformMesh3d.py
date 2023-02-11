
import numpy as np
from types import ModuleType
from scipy.sparse import csr_matrix, diags  
from .Mesh3d import Mesh3d
from .StructureMesh3dDataStructure import StructureMesh3dDataStructure

from ..geometry import project

class UniformMesh3d(Mesh3d):
    """
    @brief 三维 x 和 y 和 z 方向均匀离散的结构网格
    """
    def __init__(self, extent, 
        h=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
        itype=np.int_, ftype=np.float64):
        self.extent = extent
        self.h = h 
        self.origin = origin

        self.nx = extent[1] - extent[0]
        self.ny = extent[3] - extent[2]
        self.nz = extent[5] - extent[4]
        self.NC = self.nx * self.ny *  self.nz
        self.NN = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
        self.ds = StructureMesh3dDataStructure(self.nx, self.ny, self.nz)

        self.ftype = ftype
        self.ityep = itype
        self.meshtype = 'UniformMesh3d'

    def geo_dimension(self):
        return 3

    def top_dimension(self):
        return 3

    def number_of_nodes(self):
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        return (nx+1)*(ny+1)*(nz+1)

    @property
    def node(self):
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        box = [self.origin[0], self.origin[0] + nx*self.h[0], 
               self.origin[1], self.origin[1] + ny*self.h[1],
               self.origin[2], self.origin[2] + nz*self.h[2]]
        node = np.zeros((nx+1, ny+1, nz+1, GD), dtype=self.ftype)
        node[..., 0], node[..., 1], node[..., 2] = np.mgrid[
                box[0]:box[1]:complex(0, nx+1),
                box[2]:box[3]:complex(0, ny+1),
                box[4]:box[5]:complex(0, nz+1)]
        return node

    def entity_barycenter(self, etype='cell'):
        """
        @brief 
        """
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        if etype in {'cell', 3}:
            box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx-1)*self.h[0], 
                   self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny-1)*self.h[1], 
                   self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz-1)*self.h[2]]
            bc = np.zeros((nx, ny, nz, GD), dtype=self.ftype)
            bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                    box[0]:box[1]:complex(0, nx),
                    box[2]:box[3]:complex(0, ny),
                    box[4]:box[5]:complex(0, nz)]
            return bc

        elif etype in {'facex'}: # 法线和 x 轴平行的面
            box = [self.origin[0], self.origin[0] + nx*self.h[0],
                   self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny - 1)*self.h[1],
                   self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
            bc = np.zeros((nx + 1, ny, nz, 3), dtype=self.ftype)
            bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                                 box[0]:box[1]:complex(0, nx + 1),
                                                 box[2]:box[3]:complex(0, ny),
                                                 box[4]:box[5]:complex(0, nz)]
            return bc

        elif etype in {'facey'}: # 法线和 y 轴平行的面
            box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
                   self.origin[1], self.origin[1] + ny*self.h[1],
                   self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
            bc = np.zeros((nx, ny + 1, nz, 3), dtype=self.ftype)
            bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                                 box[0]:box[1]:complex(0, nx),
                                                 box[2]:box[3]:complex(0, ny + 1),
                                                 box[4]:box[5]:complex(0, nz)]
            return bc

        elif etype in {'facez'}: # 法线和 z 轴平行的面
            box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
                   self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny - 1)*self.h[1],
                   self.origin[2], self.origin[2] + nz*self.h[2]]
            bc = np.zeros((nx, ny, nz + 1, 3), dtype=self.ftype)
            bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                                 box[0]:box[1]:complex(0, nx),
                                                 box[2]:box[3]:complex(0, ny),
                                                 box[4]:box[5]:complex(0, nz + 1)]
            return bc

        elif etype in {'face', 2}: # 所有的面
            box = [self.origin[0], self.origin[0] + nx*self.h[0],
                   self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny - 1)*self.h[1],
                   self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
            xbc = np.zeros((nx + 1, ny, nz, 3), dtype=self.ftype)
            xbc[..., 0], xbc[..., 1], xbc[..., 2] = np.mgrid[
                                                 box[0]:box[1]:complex(0, nx + 1),
                                                 box[2]:box[3]:complex(0, ny),
                                                 box[4]:box[5]:complex(0, nz)]

            box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
                   self.origin[1], self.origin[1] + ny*self.h[1],
                   self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
            ybc = np.zeros((nx, ny + 1, nz, 3), dtype=self.ftype)
            ybc[..., 0], ybc[..., 1], ybc[..., 2] = np.mgrid[
                                                 box[0]:box[1]:complex(0, nx),
                                                 box[2]:box[3]:complex(0, ny + 1),
                                                 box[4]:box[5]:complex(0, nz)]

            box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
                   self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny - 1)*self.h[1],
                   self.origin[2], self.origin[2] + nz*self.h[2]]
            zbc = np.zeros((nx, ny, nz + 1, 3), dtype=self.ftype)
            zbc[..., 0], zbc[..., 1], zbc[..., 2] = np.mgrid[
                                                 box[0]:box[1]:complex(0, nx),
                                                 box[2]:box[3]:complex(0, ny),
                                                 box[4]:box[5]:complex(0, nz + 1)]

            return xbc, ybc, zbc

        elif etype in {'edgex'}: # 切向与 x 轴平行的边
            box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
                   self.origin[1], self.origin[1] + ny*self.h[1],
                   self.origin[2], self.origin[2] + nz*self.h[2]]
            bc = np.zeros((nx, ny + 1, nz + 1, 3), dtype=self.ftype)
            bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                     box[0]:box[1]:complex(0, nx),
                                     box[2]:box[3]:complex(0, ny + 1),
                                     box[4]:box[5]:complex(0, nz + 1)]
            return bc
        elif etype in {'edgey'}: # 切向与 y 轴平行的边
            box = [self.origin[0], self.origin[0] + nx*self.h[0],
                   self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny - 1)*self.h[1],
                   self.origin[2], self.origin[2] + nz*self.h[2]]
            bc = np.zeros((nx + 1, ny, nz + 1, 3), dtype=self.ftype)
            bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                                 box[0]:box[1]:complex(0, nx + 1),
                                                 box[2]:box[3]:complex(0, ny),
                                                 box[4]:box[5]:complex(0, nz + 1)]
            return bc
        elif etype in {'edgez'}: # 切向与 z 轴平行的边
            box = [self.origin[0], self.origin[0] + nx*self.h[0],
                   self.origin[1], self.origin[1] + ny*self.h[1],
                   self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
            bc = np.zeros((nx + 1, ny + 1, nz, 3), dtype=self.ftype)
            bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                                 box[0]:box[1]:complex(0, nx + 1),
                                                 box[2]:box[3]:complex(0, ny + 1),
                                                 box[4]:box[5]:complex(0, nz)]
            return bc
        elif etype in {'edge', 1}: # 所有的边
            box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
                   self.origin[1], self.origin[1] + ny*self.h[1],
                   self.origin[2], self.origin[2] + nz*self.h[2]]
            xbc = np.zeros((nx, ny + 1, nz + 1, 3), dtype=self.ftype)
            xbc[..., 0], xbc[..., 1], xbc[..., 2] = np.mgrid[
                                                 box[0]:box[1]:complex(0, nx),
                                                 box[2]:box[3]:complex(0, ny + 1),
                                                 box[4]:box[5]:complex(0, nz + 1)]

            box = [self.origin[0], self.origin[0] + nx*self.h[0],
                   self.origin[1], self.origin[1] + ny*self.h[1],
                   self.origin[2] + self.h[2] / 2, self.origin[2] + (nz - 1)*self.h[2]]
            ybc = np.zeros((nx + 1, ny + 1, nz, 3), dtype=self.ftype)
            ybc[..., 0], ybc[..., 1], ybc[..., 2] = np.mgrid[
                                                 box[0]:box[1]:complex(0, nx + 1),
                                                 box[2]:box[3]:complex(0, ny + 1),
                                                 box[4]:box[5]:complex(0, nz)]

            box = [self.origin[0], self.origin[0] + nx*self.h[0],
                   self.origin[1], self.origin[1] + ny*self.h[1],
                   self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
            zbc = np.zeros((nx + 1, ny + 1, nz, 3), dtype=self.ftype)
            zbc[..., 0], zbc[..., 1], zbc[..., 2] = np.mgrid[
                                                 box[0]:box[1]:complex(0, nx + 1),
                                                 box[2]:box[3]:complex(0, ny + 1),
                                                 box[4]:box[5]:complex(0, nz)]

            return xbc, ybc, zbc
        elif etype in {'node', 0}:
            return node
        else:
            raise ValueError('the entity type `{}` is not correct!'.format(etype)) 

    def cell_volume(self):
        """
        @brief 返回单元的体积，注意这里只返回一个值（因为所有单元体积相同）
        """
        return self.h[0]*self.h[1]*self.h[2]

    def face_area(self):
        """
        @brief 返回面的面积，注意这里返回三个值
        """
        return self.h[1]*self.h[2], self.h[0]*self.h[2], self.h[0]*self.h[1]

    def edge_length(self):
        """
        @brief 返回边长，注意这里返回两个值，一个 x 方向，一个 y 方向
        """
        return self.h

    def function(self, etype='node', dtype=None, ex=0):
        """
        @brief 返回定义在节点、网格边、或者网格单元上离散函数（数组），元素取值为0

        @param[in] ex 非负整数，把离散函数向外扩展一定宽度  
        """

        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        dtype = self.ftype if dtype is None else dtype
        if etype in {'node', 0}:
            uh = np.zeros((nx+1+2*ex, ny+1+2*ex, nz+1+2*ex), dtype=dtype)
        elif etype in {'facex'}: # 法线和 x 轴平行的面
            uh = np.zeros((nx+1, ny, nz), dtype=dtype)
        elif etype in {'facey'}: # 法线和 y 轴平行的面
            uh = np.zeros((nx, ny+1, nz), dtype=dtype)
        elif etype in {'facez'}: # 法线和 z 轴平行的面
            uh = np.zeros((nx, ny, nz+1), dtype=dtype)
        elif etype in {'face', 2}: # 所有的面
            ex = np.zeros((nx+1, ny, nz), dtype=dtype)
            ey = np.zeros((nx, ny+1, nz), dtype=dtype)
            ez = np.zeros((nx, ny, nz+1), dtype=dtype)
            uh = (ex, ey, ez)
        elif etype in {'edgex'}: # 切向与 x 轴平行的边
            uh = np.zeros((nx, ny+1, nz+1), dtype=dtype)
        elif etype in {'edgey'}: # 切向与 y 轴平行的边
            uh = np.zeros((nx+1, ny, nz+1), dtype=dtype)
        elif etype in {'edgez'}: # 切向与 z 轴平行的边
            uh = np.zeros((nx+1, ny+1, nz), dtype=dtype)
        elif etype in {'edge', 1}: # 所有的边 
            ex = np.zeros((nx, ny+1, nz+1), dtype=dtype)
            ey = np.zeros((nx+1, ny, nz+1), dtype=dtype)
            ez = np.zeros((nx+1, ny+1, nz), dtype=dtype)
            uh = (ex, ey, ez)
        elif etype in {'cell', 2}:
            uh = np.zeros((nx+2*ex, ny+2*ex, nz+2*ex), dtype=dtype)
        else:
            raise ValueError('the entity `{}` is not correct!'.format(entity)) 

        return uh

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

    def value(self, p, f):
        """
        @brief 根据已知网格节点上的值，构造函数，求出非网格节点处的值

        f: (nx+1, ny+1)
        """
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        box = [self.origin[0], self.origin[0] + nx*self.h[0], 
               self.origin[1], self.origin[1] + ny*self.h[1],
               self.origin[2], self.origin[2] + nz*self.h[2]]

        hx = self.h[0]
        hy = self.h[1]     
        hz = self.h[2]  
        
        i, j, k = self.cell_location(p)
        i[i==nx] = i[i==nx]-1
        j[j==ny] = j[j==ny]-1
        k[k==nz] = k[k==nz]-1
        x0 = i*hx+box[0]
        y0 = j*hy+box[2]
        z0 = k*hz+box[4]
        a = (p[..., 0]-x0)/hx
        b = (p[..., 1]-y0)/hy
        c = (p[..., 2]-z0)/hz
        d = (x0+hx-p[...,0])/hx
        e = (y0+hy-p[...,1])/hy
        f = (z0+hy-p[...,2])/hz
        F = f[i, j, k]*(1-a)*(1-b)*(1-c)\
	  + f[i+1, j, k]*(1-d)*(1-b)*(1-c)\
	  + f[i, j+1, k]*(1-a)*(1-e)*(1-c)\
	  + f[i+1, j+1, k]*(1-d)*(1-e)*(1-c)\
	  + f[i, j, k+1]*(1-a)*(1-b)*(1-f)\
	  + f[i+1, j, k+1]*(1-d)*(1-b)*(1-f)\
	  + f[i, j+1, k+1]*(1-a)*(1-e)*(1-f)\
	  + f[i+1, j+1, k+1]*(1-d)*(1-e)*(1-f)
        return F
        
    def interpolation(self, f, intertype='node'):
        node = self.node
        if intertype == 'node':
            F = f(node)
        elif intertype in {'facex'}: # 法线和 x 轴平行的面
            xbc = self.entity_barycenter('facex')
            F = f(xbc)
        elif intertype in {'facey'}: # 法线和 y 轴平行的面
            ybc = self.entity_barycenter('facey')
            F = f(ybc)
        elif intertype in {'facez'}: # 法线和 z 轴平行的面
            zbc = self.entity_barycenter('facez')
            F = f(zbc)
        elif intertype in {'face', 2}: # 所有的面
            xbc, ybc, zbc = self.entity_barycenter('face')
            F = f(xbc), f(ybc), f(zbc)
        elif intertype in {'edgex'}: # 切向与 x 轴平行的边
            xbc = self.entity_barycenter('edgex')
            F = f(xbc)
        elif intertype in {'edgey'}: # 切向与 y 轴平行的边
            ybc = self.entity_barycenter('edgey')
            F = f(ybc)
        elif intertype in {'edgez'}: # 切向与 z 轴平行的边
            zbc = self.entity_barycenter('edgez')
            F = f(zbc)
        elif intertype in {'edge', 1}: # 所有的边
            xbc, ybc, zbc = self.entity_barycenter('edge')
            F = f(xbc), f(ybc), f(zbc)
        elif intertype == 'cell':
            bc = self.entity_barycenter('cell')
            F = f(bc)
        return F

    def gradient(self, f, order=1):
        """
        @brief 求网格函数 f 的梯度
        """
        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]
        fx, fy, fz= np.gradient(f, hx, hy, hz, edge_order=order)
        return fx, fy, fz
        
    def div(self, f_x, f_y, f_z, order=1):
        """
        @brief 求向量网格函数 (fx, fy) 的散度
        """

        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]
        f_xx, f_xy, f_xz = np.gradient(f_x, hx, hy, hz, edge_order=order)
        f_yx, f_yy, f_yz = np.gradient(f_y, hx, hy, hz, edge_order=order)
        f_zx, f_zy, f_zz = np.gradient(f_z, hx, hy, hz, edge_order=order)
        return f_xx + f_yy + f_zz

    def laplace(self, f, order=1):
        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]
        fx, fy, fz = np.gradient(f, hx, hy, hz, edge_order=order)
        fxx, fxy, fxz = np.gradient(fx, hx, hy, hz, edge_order=order)
        fyx, fyy, fyz = np.gradient(fy, hx, hy, hz, edge_order=order)
        fzx, fzy, fzz = np.gradient(fz, hx, hy ,hz, edge_order=order)
        return fxx + fyy + fzz

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

        A = diags([2 * (cx + cy + cz)], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(-cx, (NN - n1 * n2,))
        I = k[1:, :, :].flat
        J = k[0:-1, :, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-cy, (NN - n0 * n2,))
        I = k[:, 1:, :].flat
        J = k[:, 0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-cz, (NN - n0 * n1,))
        I = k[:, :, 1:].flat
        J = k[:, :, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A

    def mass_matrix(self):
        h = self.h
        Mc = np.array([[8., 4., 2., 4., 4., 2., 1., 2.],
                       [4., 8., 4., 2., 2., 4., 2., 1.],
                       [2., 4., 8., 4., 1., 2., 4., 2.],
                       [4., 2., 4., 8., 2., 1., 2., 4.],
                       [4., 2., 1., 2., 8., 4., 2., 4.],
                       [2., 4., 2., 1., 4., 8., 4., 2.],
                       [1., 2., 4., 2., 2., 4., 8., 4.],
                       [2., 1., 2., 4., 4., 2., 4., 8.]])*(h[0]*h[1]*h[2]/36/6)
        cell2node = self.entity('cell')
        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        data = np.broadcast_to(Mc, (NC, 8, 8))
        I = np.broadcast_to(cell2node[..., None], (NC, 8, 8))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 8, 8))
        M = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return M

    def stiff_matrix(self):
        h = self.h
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
        cell2node = self.entity('cell')
        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        data = np.broadcast_to(Sc, (NC, 8, 8))
        I = np.broadcast_to(cell2node[..., None], (NC, 8, 8))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 8, 8))
        S = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return S

    def grad_2_matrix(self):
        h = self.h
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
        cell2node = self.entity('cell')
        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        data = np.broadcast_to(N2c, (NC, 8, 8))
        I = np.broadcast_to(cell2node[..., None], (NC, 8, 8))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 8, 8))
        N2 = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return N2

    def grad_jump_matrix(self):
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        h, nx, ny, nz = self.h, self.ds.nx, self.ds.ny, self.ds.nz

        face = self.entity('face')
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
        cellvol = self.cell_volume()
        cell2node = self.entity('cell')
        cellbar = self.entity_barycenter('cell')

        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        node = self.entity('node')
        cell = self.entity('cell')

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
        h, origin, nx, ny, nz = self.h, self.origin, self.ds.nx, self.ds.ny, self.ds.nz
        cell = self.entity('cell').reshape(nx, ny, nz, 8)

        NS = len(point) 
        NN = self.number_of_nodes()

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
        print(I.shape)
        print(J.shape)
        print(data.shape)

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
        return UniformMesh3dFunction(self, f)

    def show_function(self, plot, uh, cmap='jet'):
        """
        @brief 显示一个定义在网格节点上的函数
        """
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            axes = fig.add_subplot(111, projection='3d')
        else:
            axes = plot
        node = self.node
        return axes.plot_surface(node[..., 0], node[..., 1], uh, cmap=cmap)


    def show_animation(self, 
            fig, axes, box, init, forward, 
            fname='test.mp4',
            fargs=None, frames=1000, lw=2, interval=50):
        import matplotlib.animation as animation

        data = init(axes)
        def func(n, *fargs):
            Ez, t = forward(n)
            data.set_data(Ez)
            s = "frame=%05d, time=%0.8f"%(n, t)
            print(s)
            axes.set_title(s)
            #fig.colorbar(data)
            return data 

        ani = animation.FuncAnimation(fig, func, frames=frames, interval=interval)
        ani.save(fname)

    def boundary_node_flag(self):
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        shape = (nx+1, ny+1, nz+1)
        isBdNode = np.ones(shape, dtype=np.bool)
        isBdNode[1:-1, 1:-1, 1:-1] = False
        return isBdNode.flat

    def cell_location(self, p):
        """
        @brief 给定一组点，确定所有点所在的单元

        """
        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz

        v = p - np.array(self.origin, dtype=self.ftype)
        n0 = v[..., 0]//hx
        n1 = v[..., 1]//hy
        n2 = v[..., 2]//hz

        return n0.astype('int64'), n1.astype('int64'), n2.astype('int64')

    def to_vtk_file(self, filename, celldata=None, nodedata=None):
        """
        @brief 输出为 vtk 数据格式

        """
        from pyevtk.hl import gridToVTK

        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        box = [self.origin[0], self.origin[0] + nx*self.h[0], 
               self.origin[1], self.origin[1] + ny*self.h[1],
               self.origin[2], self.origin[2] + ny*self.h[2],
               ]

        x = np.linspace(box[0], box[1], nx+1)
        y = np.linspace(box[2], box[3], ny+1)
        z = np.linspace(box[4], box[5], nz+1)
        gridToVTK(filename, x, y, z, cellData=celldata, pointData=nodedata)

        return filename
        
        
        
class UniformMesh3dFunction():
    def __init__(self, mesh, f):
        self.mesh = mesh # (nx+1, ny+1, nz+1)
        self.f = f   # (nx+1, ny+1, nz+1)
        self.fx, self.fy ,self.fz = mesh.gradient(f) 

    def __call__(self, p):
        mesh = self.mesh
        F = mesh.value(p, self.f)
        return F
    
    def value(self, p):
        mesh = self.mesh
        F = mesh.value(p, self.f)
        return F

    def gradient(self, p):
        mesh = self.mesh
        fx = self.fx
        fy = self.fy
        fz = self.fz
        gf = np.zeros_like(p)
        gf[..., 0] = mesh.value(p, fx)
        gf[..., 1] = mesh.value(p, fy)
        gf[..., 2] = mesh.value(p, fz)
        return gf
        
    def project(self, p):
        """
        @brief 把曲线附近的点投影到曲线上
        """
        p, d = project(self, p, maxit=200, tol=1e-8, returnd=True)
        return p, d 
