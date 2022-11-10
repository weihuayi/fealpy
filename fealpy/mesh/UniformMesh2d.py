
import numpy as np
from types import ModuleType
from scipy.sparse import coo_matrix, csr_matrix
from .Mesh2d import Mesh2d
from .StructureMesh2dDataStructure import StructureMesh2dDataStructure

from ..geometry import project

"""
二维 x 和 y 方向均匀离散的结构网格
"""

class UniformMesh2d(Mesh2d):
    def __init__(self, extent, 
            h=(1.0, 1.0), origin=(0.0, 0.0),
            itype=np.int_, ftype=np.float64):
        self.extent = extent
        self.h = h 
        self.origin = origin

        nx = extent[1] - extent[0]
        ny = extent[3] - extent[2]
        self.ds = StructureMesh2dDataStructure(nx, ny,itype = itype)

        self.itype = itype 
        self.ftype = ftype 
        self.meshtype = 'StructureQuadMesh2d'

    def geo_dimension(self):
        return 2

    def top_dimension(self):
        return 2

    @property
    def node(self):
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        box = [self.origin[0], self.origin[0] + nx*self.h[0], 
               self.origin[1], self.origin[1] + ny*self.h[1]]
        node = np.zeros((nx+1, ny+1, GD), dtype=self.ftype)
        node[..., 0], node[..., 1] = np.mgrid[
                box[0]:box[1]:complex(0, nx+1),
                box[2]:box[3]:complex(0, ny+1)]
        return node

    def entity_barycenter(self, etype=2):
        """
        @brief 
        """
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        if etype in {'cell', 2}:
            box = [self.origin[0] + self.h[0]/2, self.origin[0] + (nx-1)*self.h[0], 
                   self.origin[1] + self.h[1]/2, self.origin[1] + (ny-1)*self.h[1]]
            bc = np.zeros((nx, ny, 2), dtype=self.ftype)
            bc[..., 0], bc[..., 1] = np.mgrid[
                    box[0]:box[1]:complex(0, nx),
                    box[2]:box[3]:complex(0, ny)]
            return bc
        elif etype in {'edge', 'face', 1}:

            box = [self.origin[0] + self.h[0]/2, self.origin[0] + (nx-1)*self.h[0],
                   self.origin[1],               self.origin[1] + ny*self.h[1]]
            xbc = np.zeros((nx, ny+1, 2), dtype=self.ftype)
            xbc[..., 0], xbc[..., 1] = np.mgrid[
                    box[0]:box[1]:complex(0, nx),
                    box[2]:box[3]:complex(0, ny+1)]

            box = [self.origin[0],               self.origin[0] + nx*self.h[0],
                   self.origin[1] + self.h[1]/2, self.origin[1] + (ny-1)*self.h[1]]
            ybc = np.zeros((nx+1, ny, 2), dtype=self.ftype)
            ybc[..., 0], ybc[..., 1] = np.mgrid[
                    box[0]:box[1]:complex(0, nx+1),
                    box[2]:box[3]:complex(0, ny)]
            return xbc, ybc 

        elif etype in {'edgex'}:
            box = [self.origin[0] + self.h[0]/2, self.origin[0] + (nx-1)*self.h[0],
                   self.origin[1],               self.origin[1] + ny*self.h[1]]
            bc = np.zeros((nx, ny+1, 2), dtype=self.ftype)
            bc[..., 0], bc[..., 1] = np.mgrid[
                    box[0]:box[1]:complex(0, nx),
                    box[2]:box[3]:complex(0, ny+1)]
            return bc

        elif etype in {'edgey'}:
            box = [self.origin[0],               self.origin[0] + nx*self.h[0],
                   self.origin[1] + self.h[1]/2, self.origin[1] + (ny-1)*self.h[1]]
            bc = np.zeros((nx+1, ny, 2), dtype=self.ftype)
            bc[..., 0], bc[..., 1] = np.mgrid[
                    box[0]:box[1]:complex(0, nx+1),
                    box[2]:box[3]:complex(0, ny)]
            return bc
        elif etype in {'node', 0}:
            return node
        else:
            raise ValueError('the entity type `{}` is not correct!'.format(etype)) 

    def cell_area(self):
        """
        @brief 返回单元的面积，注意这里只返回一个值（因为所有单元面积相同）
        """
        return self.h[0]*self.h[1]

    def edge_length(self, index=np.s_[:]):
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
            uh = np.zeros((nx+1+2*ex, ny+1+2*ex), dtype=dtype)
        elif etype in {'edge', 1}:
            ex = np.zeros((nx, ny+1), dtype=dtype)
            ey = np.zeros((nx+1, ny), dtype=dtype)
            uh = (ex, ey)
        elif etype in {'edgex'}:
            uh = np.zeros((nx, ny+1), dtype=dtype)
        elif etype in {'edgey'}:
            uh = np.zeros((nx+1, ny), dtype=dtype)
        elif etype in {'cell', 2}:
            uh = np.zeros((nx+2*ex, ny+2*ex), dtype=dtype)
        else:
            raise ValueError('the entity `{}` is not correct!'.format(entity)) 

        return uh

    def data_edge_to_cell(self, Ex, Ey):
        """
        @brief 把定义在边上的数组转换到单元上
        """
        dx = self.function(etype='cell')
        dy = self.function(etype='cell')

        dx[:] = (Ex[:, :-1] + Ex[:, 1:])/2.0
        dy[:] = (Ey[:-1, :] + Ey[1:, :])/2.0

        return dx, dy

    def mass_matrix(self):
        h = self.h
        Mc = np.array([[4., 2., 1., 2.],
                      [2., 4., 2., 1.],
                      [1., 2., 4., 2.],
                      [2., 1., 2., 4.]], dtype=np.float_)*h[0]*h[1]/36
        cell2node = self.entity('cell')
        NC = self.number_of_cells()

        data = np.broadcast_to(Mc, (NC, 4, 4))
        I = np.broadcast_to(cell2node[..., None], (NC, 4, 4))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 4, 4))
        M = csr_matrix((data, (I, J)), shape=(NN, NN))
        return M

    def value(self, p, f):
        """
        @brief 根据已知网格节点上的值，构造函数，求出非网格节点处的值

        f: (nx+1, ny+1)
        """
        nx = self.ds.nx
        ny = self.ds.ny
        box = [self.origin[0], self.origin[0] + nx*self.h[0], 
               self.origin[1], self.origin[1] + ny*self.h[1]]

        hx = self.h[0]
        hy = self.h[1]       
        
        i, j = self.cell_location(p)
        x0 = i*hx+box[0]
        y0 = j*hy+box[2]
        F = f[i,j]*(1-(p[...,0]-x0)/hx)*(1-(p[...,1]-y0)/hy)\
          + f[i+1,j]*(1-(x0+hx-p[...,0])/hx)*(1-(p[...,1]-y0)/hy)\
          + f[i,j+1]*(1-(p[...,0]-x0)/hx)*(1-(y0+hy-p[...,1])/hy)\
          + f[i+1,j+1]*(1-(x0+hx-p[...,0])/hx)*(1-(y0+hy-p[...,1])/hy)
        return F
        

    def interpolation(self, f, intertype='node'):
        nx = self.ds.nx
        ny = self.ds.ny
        node = self.node
        if intertype == 'node':
            F = f(node)
        elif intertype == 'edge':
            xbc, ybc = self.entity_barycenter('edge')
            F = f(xbc), f(ybc)
        elif intertype == 'edgex':
            xbc = self.entity_barycenter('edgex')
            F = f(xbc)
        elif intertype == 'edgey':
            ybc = self.entity_barycenter('edgey')
            F = f(ybc)
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
        fx, fy = np.gradient(f, hx, hy, edge_order=order)
        return fx, fy
        
    def div(self, f_x, f_y, order=1):
        """
        @brief 求向量网格函数 (fx, fy) 的散度
        """

        hx = self.h[0]
        hy = self.h[1]
        f_xx,f_xy = np.gradient(f_x, hx, edge_order=order)
        f_yx,f_yy = np.gradient(f_y, hy, edge_order=order)
        return fxx + fyy

    def laplace(self, f, order=1):
        hx = self.h[0]
        hy = self.h[1]
        fx, fy = np.gradient(f, hx, hy, edge_order=order)
        fxx,fxy = np.gradient(fx, hx, edge_order=order)
        fyx,fyy = np.gradient(fy, hy, edge_order=order)
        return fxx + fyy 

    def laplace_operator(self):
        """
        @brief 构造笛卡尔网格上的 Laplace 离散算子，其中 x 方向和 y
        方向都均匀剖分，但步长可以不一样
        """

        n0 = self.ds.nx + 1
        n1 = self.ds.ny + 1
        cx = 1/(self.hx**2)
        cy = 1/(self.hy**2)
        NN = self.number_of_nodes()
        k = np.arange(NN).reshape(n0, n1)

        A = diags([2*(cx+cy)], [0], shape=(NN, NN), format='coo')

        val = np.broadcast_to(-cx, (NN-n1, ))
        I = k[1:, :].flat
        J = k[0:-1, :].flat
        A += coo_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += coo_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-cy, (NN-n0, ))
        I = k[:, 1:].flat
        J = k[:, 0:-1].flat
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
            data.set_array(Ez)
            s = "frame=%05d, time=%0.8f"%(n, t)
            print(s)
            axes.set_title(s)
            axes.set_aspect('equal')
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
        nx = self.ds.nx
        ny = self.ds.ny

        v = p - np.array(self.origin, dtype=self.ftype)
        n0 = v[..., 0]//hx
        n1 = v[..., 1]//hy

        return n0.astype('int64'), n1.astype('int64')

    def to_vtk_file(self, filename, celldata=None, nodedata=None):
        """


        """
        from pyevtk.hl import gridToVTK

        nx = self.ds.nx
        ny = self.ds.ny
        box = [self.origin[0], self.origin[0] + nx*self.h[0], 
               self.origin[1], self.origin[1] + ny*self.h[1]]
	
        x = np.linspace(box[0], box[1], nx+1)
        y = np.linspace(box[2], box[3], ny+1)
        z = np.zeros(1)
        gridToVTK(filename, x, y, z, cellData=celldata, pointData=nodedata)

        return filename

    def fast_sweeping_method(self):
        """
        @brief 均匀网格上的 fast sweeping method

        @note 注意，我们这里假设 x 和 y 方向剖分的段数相等
        """
        a = np.zeros(ns+1, dtype=np.float64) 
        b = np.zeros(ns+1, dtype=np.float64)
        c = np.zeros(ns+1, dtype=np.float64)

        n = 0
        for i in range(1, ns+2):
            a[:] = np.minimum(phi[i-1, 1:-1], phi[i+1, 1:-1])
            b[:] = np.minimum(phi[i, 0:ns+1], phi[i, 2:])
            flag = np.abs(a-b) >= h 
            c[flag] = np.minimum(a[flag], b[flag]) + h 
            c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
            phi[i, 1:-1] = np.minimum(c, phi[i, 1:-1])

            fname = output + 'test'+ str(n).zfill(10)
            data = (sign*phi[1:-1, 1:-1]).reshape(ns+1, ns+1, 1)
            nodedata = {'phi':data}
            mesh.to_vtk_file(fname, nodedata=nodedata)
            n += 1


        for i in range(ns+1, 0, -1):
            a[:] = np.minimum(phi[i-1, 1:-1], phi[i+1, 1:-1])
            b[:] = np.minimum(phi[i, 0:ns+1], phi[i, 2:])
            flag = np.abs(a-b) >= h 
            c[flag] = np.minimum(a[flag], b[flag]) + h 
            c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
            phi[i, 1:-1] = np.minimum(c, phi[i, 1:-1])

            fname = output + 'test'+ str(n).zfill(10)
            data = (sign*phi[1:-1, 1:-1]).reshape(ns+1, ns+1, 1)
            nodedata = {'phi':data}
            mesh.to_vtk_file(fname, nodedata=nodedata)
            n += 1

        for j in range(1, ns+2):
            a[:] = np.minimum(phi[0:ns+1, j], phi[2:, j])
            b[:] = np.minimum(phi[1:-1, j-1], phi[1:-1, j+1])
            flag = np.abs(a-b) >= h 
            c[flag] = np.minimum(a[flag], b[flag]) + h 
            c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
            phi[1:-1, j] = np.minimum(c, phi[1:-1, j])

            fname = output + 'test'+ str(n).zfill(10)
            data = (sign*phi[1:-1, 1:-1]).reshape(ns+1, ns+1, 1)
            nodedata = {'phi':data}
            mesh.to_vtk_file(fname, nodedata=nodedata)
            n += 1

        for j in range(ns+1, 0, -1):
            a[:] = np.minimum(phi[0:ns+1, j], phi[2:, j])
            b[:] = np.minimum(phi[1:-1, j-1], phi[1:-1, j+1])
            flag = np.abs(a-b) >= h 
            c[flag] = np.minimum(a[flag], b[flag]) + h 
            c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
            phi[1:-1, j] = np.minimum(c, phi[1:-1, j])

            fname = output + 'test'+ str(n).zfill(10)
            data = (sign*phi[1:-1, 1:-1]).reshape(ns+1, ns+1, 1)
            nodedata = {'phi':data}
            mesh.to_vtk_file(fname, nodedata=nodedata)
            n += 1



class UniformMesh2dFunction():
    def __init__(self, mesh, f):
        self.mesh = mesh # (nx+1, ny+1)
        self.f = f   # (nx+1, ny+1)
        self.fx, self.fy = mesh.gradient(f) 

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
        gf = np.zeros_like(p)
        gf[..., 0] = mesh.value(p, fx)
        gf[..., 1] = mesh.value(p, fy)
        return gf
        
    def project(self, p):
        """
        @brief 把曲线附近的点投影到曲线上
        """
        p, d = project(self, p, maxit=200, tol=1e-8, returnd=True)
        return p, d 
         






