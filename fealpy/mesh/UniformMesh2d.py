
import numpy as np
from types import ModuleType
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from .Mesh2d import Mesh2d
from .StructureMesh2dDataStructure import StructureMesh2dDataStructure

from ..geometry import project


class UniformMesh2d(Mesh2d):
    """
    @brief 二维 x 和 y 方向均匀离散的结构网格
    """
    def __init__(self, extent, 
            h=(1.0, 1.0), origin=(0.0, 0.0),
            itype=np.int_, ftype=np.float64):
        self.extent = extent
        self.h = h 
        self.origin = origin

        self.nx = self.extent[1] - self.extent[0]
        self.ny = self.extent[3] - self.extent[2]
        self.NC = self.nx * self.ny
        self.NN = (self.nx + 1) * (self.ny + 1)
        self.ds = StructureMesh2dDataStructure(self.nx, self.ny, itype=itype)

        self.itype = itype 
        self.ftype = ftype 
        self.meshtype = 'StructureQuadMesh2d'

    def uniform_refine(self, n=1, returnim=False):
        if returnim:
            nodeImatrix = []
        for i in range(n):
            print('h1', self.h[0])
            self.extent = [i * 2 for i in self.extent]
            self.h = [i / 2 for i in self.h]
            self.nx = self.extent[1] - self.extent[0]
            self.ny = self.extent[3] - self.extent[2]

            self.NC = self.nx * self.ny
            self.NN = (self.nx + 1) * (self.ny + 1)
            self.ds = StructureMesh2dDataStructure(self.nx, self.ny, itype=self.itype)

            if returnim:
                A = self.interpolation_matrix()
                nodeImatrix.append(A)

        if returnim:
            return nodeImatrix

    def geo_dimension(self):
        return 2

    def top_dimension(self):
        return 2

    @property
    def node(self):
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        box = [self.origin[0], self.origin[0] + nx * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]
        node = np.zeros((nx + 1, ny + 1, GD), dtype=self.ftype)
        node[..., 0], node[..., 1] = np.mgrid[
                                     box[0]:box[1]:complex(0, nx + 1),
                                     box[2]:box[3]:complex(0, ny + 1)]
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

    def error(self, h, nx, ny, u, uh):
        """
        @brief 计算真解在网格点处与数值解的误差

        @param[in] u
        @param[in] uh
        """
        e = u - uh

        emax = np.max(np.abs(e))
        e0 = np.sqrt(h ** 2 * np.sum(e ** 2))

        el2 = np.sqrt(1 / ((nx - 1) * (ny - 1)) * np.sum(e ** 2))

        return emax, e0, el2

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

    def function_remap(self, tmesh, p=2):
        """
        @brief 获取结构三角形和结构四边形网格上的自由度映射关系

        @example 
        phi.flat[idxMap] = uh
        """
        nx = self.ds.nx
        ny = self.ds.ny

        NN = self.number_of_nodes()
        idxMap = np.arange(NN)

        idx = np.arange(0, nx*(ny+1), 2*(ny+1)).reshape(-1, 1) + np.arange(0,
                ny+1, 2)
        idxMap[idx] = range(tmesh.number_of_nodes())

        return idxMap

    def mass_matrix(self):
        h = self.h
        Mc = np.array([[4., 2., 2., 1.],
                       [2., 4., 1., 2.],
                       [2., 1., 4., 2.],
                       [1., 2., 2., 4.]], dtype=np.float_)*h[0]*h[1]/36
        cell2node = self.entity('cell')
        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        data = np.broadcast_to(Mc, (NC, 4, 4))
        I = np.broadcast_to(cell2node[..., None], (NC, 4, 4))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 4, 4))
        M = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return M

    def stiff_matrix(self):
        h = self.h
        S0c = np.array([[ 2.,  1., -2., -1.],
                        [ 1.,  2., -1., -2.],
                        [-2., -1.,  2.,  1.],
                        [-1., -2.,  1.,  2.]], dtype=np.float_)*h[1]/h[0]/6
        S1c = np.array([[ 2., -2.,  1., -1.],
                        [-2.,  2., -1.,  1.],
                        [ 1., -1.,  2., -2.],
                        [-1.,  1., -2.,  2.]], dtype=np.float_)*h[0]/h[1]/6
        Sc = S0c + S1c
        cell2node = self.entity('cell')
        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        data = np.broadcast_to(Sc, (NC, 4, 4))
        I = np.broadcast_to(cell2node[..., None], (NC, 4, 4))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 4, 4))
        S = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return S

    def nabla_2_matrix(self):
        h = self.h
        N2c = np.array([[ 1., -1., -1.,  1.],
                        [-1.,  1.,  1., -1.],
                        [-1.,  1.,  1., -1.],
                        [ 1., -1., -1.,  1.]], dtype=np.float_)*4/(h[1]*h[0])
        cell2node = self.entity('cell')
        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        data = np.broadcast_to(N2c, (NC, 4, 4))
        I = np.broadcast_to(cell2node[..., None], (NC, 4, 4))
        J = np.broadcast_to(cell2node[:, None, :], (NC, 4, 4))
        N2 = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return N2

    def nabla_jump_matrix(self):
        h, nx, ny = self.h, self.ds.nx, self.ds.ny
        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        Jumpe = np.array([[ 2.,  1., -4., -2.,  2.,  1.],
                          [ 1.,  2., -2., -4.,  1.,  2.],
                          [-4., -2.,  8.,  4., -4., -2.],
                          [-2., -4.,  4.,  8., -2., -4.],
                          [ 2.,  1., -4., -2.,  2.,  1.],
                          [ 1.,  2., -2., -4.,  1.,  2.]])*(h[1]/h[0]/h[0]/6)
        edge = self.entity('edge')
        edgex = edge[:nx*(ny+1)].reshape(nx, ny+1, 2)
        edgey = edge[nx*(ny+1):].reshape(nx+1, ny, 2)

        edgey2dof = np.zeros([nx-1, ny, 6], dtype=np.int_)
        edgey2dof[..., 0:2] = edgey[1:-1]-ny-1
        edgey2dof[..., 2:4] = edgey[1:-1]
        edgey2dof[..., 4:6] = edgey[1:-1]+ny+1

        data = np.broadcast_to(Jumpe, (nx-1, ny, 6, 6))
        I = np.broadcast_to(edgey2dof[..., None], data.shape)
        J = np.broadcast_to(edgey2dof[..., None, :], data.shape)
        Jump = csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))

        edgex2dof = np.zeros([nx, ny-1, 6], dtype=np.int_)
        edgex2dof[..., 0:2] = edgex[:, 1:-1]-1
        edgex2dof[..., 2:4] = edgex[:, 1:-1]
        edgex2dof[..., 4:6] = edgex[:, 1:-1]+1

        data = np.broadcast_to(Jumpe, (nx, ny-1, 6, 6))
        I = np.broadcast_to(edgex2dof[..., None], data.shape)
        J = np.broadcast_to(edgex2dof[..., None, :], data.shape)
        Jump += csr_matrix((data.flat, (I.flat, J.flat)), shape=(NN, NN))
        return Jump

    def source_vector(self, f):
        cellarea = self.cell_area()
        cell2node = self.entity('cell')
        cellbar = self.entity_barycenter('cell')

        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        node = self.entity('node')
        cell = self.entity('cell')

        #fval = f(node[cell])*cellarea/4 # (NC, )

        fval = f(cellbar).reshape(-1) # (NC, )
        fval = fval*cellarea/4
        fval = np.broadcast_to(fval[:, None], (NC, 4))

        F = np.zeros(NN, dtype=np.float_)
        np.add.at(F, cell2node, fval)
        return F

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
        i[i==nx] = i[i==nx]-1
        j[j==ny] =j[j==ny]-1
        x0 = i*hx+box[0]
        y0 = j*hy+box[2]
        a = (p[...,0]-x0)/hx
        b = (p[...,1]-y0)/hy
        c = (x0+hx-p[...,0])/hx
        d = (y0+hy-p[...,1])/hy
        F = f[i,j]*(1-a)*(1-b)\
          + f[i+1,j]*(1-c)*(1-b)\
          + f[i,j+1]*(1-a)*(1-d)\
          + f[i+1,j+1]*(1-c)*(1-d)
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

    def fast_sweeping_method(self, phi0):
    	"""
        @brief 均匀网格上的 fast sweeping method
        @param[in] phi 是一个离散的水平集函数

        @note 注意，我们这里假设 x 和 y 方向剖分的段数相等
    	"""
    	m = 2
    	nx = self.ds.nx
    	ny = self.ds.ny
    	k = nx/ny
    	ns = ny
    	h = self.h[0]
    	
    	phi = self.function(ex=1)
    	isNearNode = self.function(dtype=np.bool_, ex=1)
    	
    	# 把水平集函数转化为离散的网格函数
    	node = self.entity('node')
    	phi[1:-1, 1:-1] = phi0
    	sign = np.sign(phi[1:-1, 1:-1])
    	
    	# 标记界面附近的点
    	isNearNode[1:-1, 1:-1] = np.abs(phi[1:-1, 1:-1]) < 2*h
    	lsfun = UniformMesh2dFunction(self, phi[1:-1, 1:-1])
    	_, d = lsfun.project(node[isNearNode[1:-1, 1:-1]])
    	phi[isNearNode] = np.abs(d) #界面附近的点用精确值
    	phi[~isNearNode] = m  # 其它点用一个比较大的值
    	
    	
    	
    	a = np.zeros(ns+1, dtype=np.float64)
    	b = np.zeros(ns+1, dtype=np.float64)
    	c = np.zeros(ns+1, dtype=np.float64)
    	d = np.zeros(int(k*ns+1), dtype=np.float64)
    	e = np.zeros(int(k*ns+1), dtype=np.float64)
    	f = np.zeros(int(k*ns+1), dtype=np.float64)
    	
    	
    	n = 0
    	for i in range(1, int(k*ns+2)):
    	    a[:] = np.minimum(phi[i-1, 1:-1], phi[i+1, 1:-1])
    	    b[:] = np.minimum(phi[i, 0:ns+1], phi[i, 2:])
    	    flag = np.abs(a-b) >= h
    	    c[flag] = np.minimum(a[flag], b[flag]) + h
    	    c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
    	    phi[i, 1:-1] = np.minimum(c, phi[i, 1:-1])
    	    n += 1
    	    
    	    
    	for i in range(int(k*ns+1), 0, -1):
    	    a[:] = np.minimum(phi[i-1, 1:-1], phi[i+1, 1:-1])
    	    b[:] = np.minimum(phi[i, 0:ns+1], phi[i, 2:])
    	    flag = np.abs(a-b) >= h 
    	    c[flag] = np.minimum(a[flag], b[flag]) + h
    	    c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
    	    phi[i, 1:-1] = np.minimum(c, phi[i, 1:-1])
    	    n += 1
    	    
    	    
    	for j in range(1, ns+2):
    	    d[:] = np.minimum(phi[0:int(k*ns+1), j], phi[2:, j])
    	    e[:] = np.minimum(phi[1:-1, j-1], phi[1:-1, j+1])
    	    flag = np.abs(d-e) >= h 
    	    f[flag] = np.minimum(d[flag], e[flag]) + h
    	    f[~flag] = (d[~flag] + e[~flag] + np.sqrt(2*h*h - (d[~flag] - e[~flag])**2))/2
    	    phi[1:-1, j] = np.minimum(f, phi[1:-1, j])
    	    n += 1
    	    
    	    
    	for j in range(ns+1, 0, -1):
    	    d[:] = np.minimum(phi[0:int(k*ns+1), j], phi[2:, j])
    	    e[:] = np.minimum(phi[1:-1, j-1], phi[1:-1, j+1])
    	    flag = np.abs(d-e) >= h
    	    f[flag] = np.minimum(d[flag], e[flag]) + h
    	    f[~flag] = (d[~flag] + e[~flag] + np.sqrt(2*h*h - (d[~flag] - e[~flag])**2))/2
    	    phi[1:-1, j] = np.minimum(f, phi[1:-1, j])
    	    n += 1
    	    
    	return sign*phi[1:-1, 1:-1]
        
    
        
    
    def interpolation_with_sample_points(self, x, y, alpha=[10, 0.001, 0.01, 0.1]):
        '''!
        @brief 将 x, y 插值为网格函数
        @param x : 样本点
        @param y : 样本点的值
        '''
        h, origin, nx, ny = self.h, self.origin, self.ds.nx, self.ds.ny
        cell = self.entity('cell').reshape(nx, ny, 4)

        NS = len(x) 
        NN = self.number_of_nodes()

        Xp = (x-origin)/h # (NS, 2)
        cellIdx = Xp.astype(np.int_) # 样本点所在单元
        val = Xp - cellIdx 

        I = np.repeat(np.arange(NS), 4)
        J = cell[cellIdx[:, 0], cellIdx[:, 1]]
        data = np.zeros([NS, 4], dtype=np.float_)
        data[:, 0] = (1-val[:, 0])*(1-val[:, 1])
        data[:, 1] = (1-val[:, 0])*val[:, 1]
        data[:, 2] = val[:, 0]*val[:, 1]
        data[:, 3] = val[:, 0]*(1-val[:, 1])

        A = csr_matrix((data.flat, (I, J.flat)), (NS, NN), dtype=np.float_)
        B = self.stiff_matrix()
        C = self.nabla_2_matrix()
        D = self.nabla_jump_matrix()

        S = alpha[0]*A.T@A + alpha[1]*B + alpha[2]*C + alpha[3]*D
        F = alpha[0]*A.T@y
        f = spsolve(S, F).reshape(nx+1, ny+1)
        return UniformMesh2dFunction(self, f)
    def t2sidx(self):
        """
        @brief 已知结构三角形网格点的值，将其排列到结构四边形网格上
        @example a[s2tidx] = uh
        """
        snx = self.ds.nx
        sny = self.ds.ny
        idx1= np.arange(0,sny+1,2)+np.arange(0,(sny+1)*(snx+1),2*(sny+1)).reshape(-1,1)
        idx1 = idx1.flatten()
        a = np.array([1,sny+1,sny+2])
        b = np.arange(0,sny,2).reshape(-1,1)
        c = np.append(a+b,[(sny+1)*2-1])
        e = np.arange(0,(sny+1)*snx,2*(sny+1)).reshape(-1,1)
        idx2 = (e+c).flatten()
        idx3  = np.arange((sny+1)*snx+1,(snx+1)*(sny+1),2)
        idx = np.r_[idx1,idx2,idx3]
        return idx

    def  s2tidx(self):
    	"""
    	@brief 已知结构四边形网格点的值，将其排列到结构三角形网格上
    	@example a[s2tidx] = uh
    	"""
    	tnx = int(self.ds.nx/2)
    	tny = int(self.ds.ny/2)
    	a = np.arange(tny+1)
    	b = 3*np.arange(tny).reshape(-1,1)+(tnx+1)*(tny+1)
    	idx1 = np.zeros((tnx+1,2*tny+1))#sny+1
    	idx1[:,0::2] = a+np.arange(tnx+1).reshape(-1,1)*(tny+1)
    	idx1[:,1::2] = b.flatten()+np.arange(tnx+1).reshape(-1,1)*(2*tny+1+tny)
    	idx1[-1,1::2] = np.arange((2*tnx+1)*(2*tny+1)-tny,(2*tnx+1)*(2*tny+1))
    	c = np.array([(tnx+1)*(tny+1)+1,(tnx+1)*(tny+1)+2])
    	d = np.arange(tny)*3
    	d = 3*np.arange(tny).reshape(-1,1)+c
    	e = np.append(d.flatten(),[d.flatten()[-1]+1])
    	idx2 = np.arange(tnx).reshape(-1,1)*(2*tny+1+tny)+e
    	
    	idx = np.c_[idx1[:tnx],idx2]
    	idx = np.append(idx.flatten(),[idx1[-1,:]])
    	idx = idx.astype(int)
    	return idx
    	
    	
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

    @classmethod
    def from_sample_points(self, x, y, nx=10, ny=10):
        '''!
        @param x, y : 样本点和值
        '''
        minx, miny = np.min(x[..., 0]), np.min(x[..., 1])
        maxx, maxy = np.max(x[..., 0]), np.max(x[..., 1])

        h = np.array([(maxx-minx)/nx, (maxy-miny)/ny])
        mesh = UniformMesh2d([0, nx+1, 0, ny+1], h, np.array([minx, miny])) 
        return mesh.interpolation_with_sample_points(x, y)



