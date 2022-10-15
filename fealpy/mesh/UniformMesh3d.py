
import numpy as np
from types import ModuleType
from scipy.sparse import coo_matrix, csr_matrix
from .Mesh3d import Mesh3d
from .StructureMesh3dDataStructure import StructureMesh3dDataStructure

class UniformMesh3d(Mesh3d):
    """
    @brief 
    """
    def __init__(self, extent, 
            h=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
            ftype=np.float64, itype=np.int_
            ):
        self.extent = extent
        self.h = h 
        self.origin = origin

        nx = extent[1] - extent[0]
        ny = extent[3] - extent[2]
        nz = extent[5] - extent[4]
        self.ds = StructureMesh3dDataStructure(nx, ny, nz)

        self.ftype = ftype
        self.ityep = itype

    def geo_dimension(self):
        return 3

    def top_dimension(self):
        return 3

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
            box = [self.origin[0] + self.h[0]/2, self.origin[0] + (nx-1)*self.h[0], 
                   self.origin[1] + self.h[1]/2, self.origin[1] + (ny-1)*self.h[1], 
                   self.origin[2] + self.h[2]/2, self.origin[2] + (nz-1)*self.h[2]]
            bc = np.zeros((nx, ny, nz, GD), dtype=self.ftype)
            bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                    box[0]:box[1]:complex(0, nx),
                    box[2]:box[3]:complex(0, ny),
                    box[4]:box[5]:complex(0, nz)]
            return bc
        elif etype in {'facex'}: # 法线和 x 轴平行的面
            pass
        elif etype in {'facey'}: # 法线和 y 轴平行的面
            pass
        elif etype in {'facez'}: # 法线和 z 轴平行的面
            pass
        elif etype in {'face', 2}: # 所有的面
            pass
        elif etype in {'edgex'}: # 切向与 x 轴平行的边
            pass 
        elif etype in {'edgey'}: # 切向与 y 轴平行的边
            pass
        elif etype in {'edgez'}: # 切向与 z 轴平行的边
            pass
        elif etype in {'edge', 1}: # 所有的边 
            pass
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
        dtype = self.ftype if dtype is None else dtype
        if etype in {'node', 0}:
            uh = np.zeros((nx+1+2*ex, ny+1+2*ex, nz+1+2*ex), dtype=dtype)
        elif etype in {'facex'}: # 法线和 x 轴平行的面
            pass
        elif etype in {'facey'}: # 法线和 y 轴平行的面
            pass
        elif etype in {'facez'}: # 法线和 z 轴平行的面
            pass
        elif etype in {'face', 2}: # 所有的面
            pass
        elif etype in {'edgex'}: # 切向与 x 轴平行的边
            pass 
        elif etype in {'edgey'}: # 切向与 y 轴平行的边
            pass
        elif etype in {'edgez'}: # 切向与 z 轴平行的边
            pass
        elif etype in {'edge', 1}: # 所有的边 
            pass
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


    def interpolation(self, f, intertype='node'):
        node = self.node
        if intertype == 'node':
            F = f(node)
        elif etype in {'facex'}: # 法线和 x 轴平行的面
            pass
        elif etype in {'facey'}: # 法线和 y 轴平行的面
            pass
        elif etype in {'facez'}: # 法线和 z 轴平行的面
            pass
        elif etype in {'face', 2}: # 所有的面
            pass
        elif etype in {'edgex'}: # 切向与 x 轴平行的边
            pass 
        elif etype in {'edgey'}: # 切向与 y 轴平行的边
            pass
        elif etype in {'edgez'}: # 切向与 z 轴平行的边
            pass
        elif etype in {'edge', 1}: # 所有的边 
            pass
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
        
    def div(self, fx, fy, fz, order=1):
        """
        @brief 求向量网格函数 (fx, fy) 的散度
        """

        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]
        fxx = np.gradient(fx, hx, edge_order=order)
        fyy = np.gradient(fy, hy, edge_order=order)
        fzz = np.gradient(fz, hz, edge_order=order)
        return fxx + fyy + fzz

    def laplace(self, f, order=1):
        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]
        fx, fy, fz = np.gradient(f, hx, hy, hz, edge_order=order)
        fxx= np.gradient(fx, hx, edge_order=order)
        fyy = np.gradient(fy, hy, edge_order=order)
        fzz = np.gradient(fz, hz, edge_order=order)
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

        return n0, n1, n2

    def to_vtk_file(self, filename, celldata=None, nodedata=None):
        """

        """
        from pyevtk.hl import gridToVTK

        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.da.nz
        box = [self.origin[0], self.origin[0] + nx*self.h[0], 
               self.origin[1], self.origin[1] + ny*self.h[1],
               self.origin[2], self.origin[2] + ny*self.h[2],
               ]

        x = np.linspace(box[0], box[1], nx+1)
        y = np.linspace(box[2], box[3], ny+1)
        z = np.linspace(box[4], box[5], nz+1)
        gridToVTK(filename, x, y, z, cellData=celldata, pointData=nodedata)

        return filename
