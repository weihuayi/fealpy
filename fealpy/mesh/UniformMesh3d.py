import numpy as np
import warnings
from scipy.sparse import csr_matrix, diags
from types import ModuleType
from .Mesh3d import Mesh3d
    
# 这个数据接口为有限元服务
from .StructureMesh3dDataStructure import StructureMesh3dDataStructure
from .mesh_tools import show_mesh_3d

from ..geometry import project

class UniformMesh3d(Mesh3d):
    """
    @brief A class for representing a three-dimensional structured mesh with uniform discretization in x, y, and z directions.
    """
    def __init__(self, extent, 
            h=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0),
            ftype=np.float64, itype=np.int_
            ):
        """
        @brief Initialize the 3D uniform mesh.

        @param[in] extent A tuple representing the range of the mesh in the x, y, and z directions.
        @param[in] h A tuple representing the mesh step sizes in the x, y, and z directions, default: (1.0, 1.0, 1.0).
        @param[in] origin A tuple representing the coordinates of the starting point, default: (0.0, 0.0, 0.0).
        @param[in] ftype Floating point type to be used, default: np.float64.
        @param[in] itype Integer type to be used, default: np.int_.

        @note The extent parameter defines the index range in the x, y, and z directions.
              We can define an index range starting from 0, e.g., [0, 10, 0, 10, 0, 10],
              or starting from a non-zero value, e.g., [2, 12, 3, 13, 4, 14]. The flexibility
              in the index range is mainly for handling different scenarios
              and data subsets, such as:
              - Subgrids
              - Parallel computing
              - Data cropping
              - Handling irregular data

        @example 
        from fealpy.mesh import UniformMesh3d

        I = [0, 1, 0, 1, 0, 1]
        h = (0.1, 0.1, 0.1)
        nx = int((I[1] - I[0])/h[0])
        ny = int((I[3] - I[2])/h[1])
        nz = int((I[5] - I[4])/h[2])
        mesh = UniformMesh3d([0, nx, 0, ny, 0, nz], h=h, origin=(I[0], I[2], I[4]))

        """
        # Mesh properties
        self.extent = extent
        self.h = h
        self.origin = origin

        self.ftype = ftype
        self.ityep = itype
 
        # Mesh dimensions
        self.nx = extent[1] - extent[0]
        self.ny = extent[3] - extent[2]
        self.nz = extent[5] - extent[4]
        self.NC = self.nx * self.ny * self.nz
        self.NN = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

        # Data structure for finite element computation
        self.ds = StructureMesh3dDataStructure(self.nx, self.ny, self.nz)

    ## @ingroup GeneralInterface
    def number_of_nodes(self):
        """
        @brief Get the number of nodes in the mesh.
 
        @return The number of nodes.
        """
        return self.NN
        
    ## @ingroup GeneralInterface
    def number_of_edges(self):
        """
        @brief Get the number of edges in the mesh.

        @note `edge` is the 1D entity

        @return The number of edges.
        """
        pass

    ## @ingroup GeneralInterface
    def number_of_faces(self):
        """
        @brief Get the number of faces in the mesh.

        @note `face` is the 2D entity
        
        @return The number of faces.
        """
        pass

    ## @ingroup GeneralInterface
    def number_of_cells(self):
        """
        @brief Get the number of cells in the mesh.

        @return The number of cells.
        """
        return self.NC

    ## @ingroup GeneralInterface
    def uniform_refine(self, n=1, surface=None, interface=None, returnim=False):
        """
        @brief
        """
        pass

    ## @ingroup GeneralInterface
    def cell_volume(self):
        """
        @brief 返回单元的体积，注意这里只返回一个值（因为所有单元体积相同）
        """
        return self.h[0]*self.h[1]*self.h[2]

    ## @ingroup GeneralInterface
    def face_area(self):
        """
        @brief 返回面的面积，注意这里返回三个值
        """
        return self.h[1]*self.h[2], self.h[0]*self.h[2], self.h[0]*self.h[1]

    ## @ingroup GeneralInterface
    def edge_length(self):
        """
        @brief 返回边长，注意这里返回三个值，一个 x 方向，一个 y 方向, 一个 z 方向
        """
        return self.h[0], self.h[1], self.h[2]

    ## @ingroup GeneralInterface
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

    ## @ingroup GeneralInterface
    def show_function(self, plot, uh, cmap='jet'):
        """
        """
        pass

    ## @ingroup GeneralInterface
    def show_animation(self, fig, axes, box, 
                       init, forward, fname='test.mp4',
                       fargs=None, frames=1000, lw=2, interval=50):
        """
        @brief
        """
        pass

    ## @ingroup GeneralInterface
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

    ## @ingroup FDMInterface
    @property
    def node(self):
        """
        @brief Get the coordinates of the nodes in the mesh.

        @return A NumPy array of shape (nx+1, ny+1, nz+1, 3) containing the coordinates of the nodes.

        @details This function calculates the coordinates of the nodes in the mesh based on the
        mesh's origin, step size, and the number of cells in the x and y directions.
        It returns a NumPy array with the coordinates of each node.

        """

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

    ## @ingroup FDMInterface
    def cell_barycenter(self):
        """
        @brief
        """
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx-1)*self.h[0], 
               self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny-1)*self.h[1], 
               self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz-1)*self.h[2]]
        bc = np.zeros((nx, ny, nz, GD), dtype=self.ftype)
        bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                             box[0]:box[1]:complex(0, nx),
                                             box[2]:box[3]:complex(0, ny),
                                             box[4]:box[5]:complex(0, nz)]
        return bc

    ## @ingroup FDMInterface
    def edge_barycenter(self):
        """
        @brief
        """
        xbc = self.edgex_barycenter()
        ybc = self.edgey_barycenter()
        zbc = self.edgez_barycenter()
        return xbc, ybc, zbc

    ## @ingroup FDMInterface
    def edgex_barycenter(self):
        """
        @brief
        """
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
               self.origin[1], self.origin[1] + ny*self.h[1],
               self.origin[2], self.origin[2] + nz*self.h[2]]
        bc = np.zeros((nx, ny + 1, nz + 1, 3), dtype=self.ftype)
        bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                             box[0]:box[1]:complex(0, nx),
                                             box[2]:box[3]:complex(0, ny + 1),
                                             box[4]:box[5]:complex(0, nz + 1)]
        return bc

    ## @ingroup FDMInterface
    def edgey_barycenter(self):
        """
        @brief
        """
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        box = [self.origin[0], self.origin[0] + nx*self.h[0],
               self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny - 1)*self.h[1],
               self.origin[2], self.origin[2] + nz*self.h[2]]
        bc = np.zeros((nx + 1, ny, nz + 1, 3), dtype=self.ftype)
        bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                             box[0]:box[1]:complex(0, nx + 1),
                                             box[2]:box[3]:complex(0, ny),
                                             box[4]:box[5]:complex(0, nz + 1)]
        return bc 

    ## @ingroup FDMInterface
    def edgez_barycenter(self):
        """
        @brief
        """
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        box = [self.origin[0], self.origin[0] + nx*self.h[0],
               self.origin[1], self.origin[1] + ny*self.h[1],
               self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
        bc = np.zeros((nx + 1, ny + 1, nz, 3), dtype=self.ftype)
        bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                             box[0]:box[1]:complex(0, nx + 1),
                                             box[2]:box[3]:complex(0, ny + 1),
                                             box[4]:box[5]:complex(0, nz)]
        return bc
    
    ## @ingroup FDMInterface
    def face_barycenter(self):
        """
        @brief
        """ 
        xbc = self.facex_barycenter()
        ybc = self.facey_barycenter()
        zbc = self.facez_barycenter()
        return xbc, ybc, zbc

    ## @ingroup FDMInterface
    def facex_barycenter(self):
        """
        @brief
        """
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        box = [self.origin[0],               self.origin[0] + nx*self.h[0],
               self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny - 1)*self.h[1],
               self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
        bc = np.zeros((nx + 1, ny, nz, 3), dtype=self.ftype)
        bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                             box[0]:box[1]:complex(0, nx + 1),
                                             box[2]:box[3]:complex(0, ny),
                                             box[4]:box[5]:complex(0, nz)]
        return bc

    ## @ingroup FDMInterface
    def facey_barycenter(self):
        """
        @brief
        """
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
               self.origin[1],               self.origin[1] + ny*self.h[1],
               self.origin[2] + self.h[2]/2, self.origin[2] + self.h[2]/2 + (nz - 1)*self.h[2]]
        bc = np.zeros((nx, ny + 1, nz, 3), dtype=self.ftype)
        bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                             box[0]:box[1]:complex(0, nx),
                                             box[2]:box[3]:complex(0, ny + 1),
                                             box[4]:box[5]:complex(0, nz)]
        return bc
    
    ## @ingroup FDMInterface
    def facez_barycenter(self):
        """
        @brief
        """
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx - 1)*self.h[0],
               self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny - 1)*self.h[1],
               self.origin[2], self.origin[2] + nz*self.h[2]]
        bc = np.zeros((nx, ny, nz + 1, 3), dtype=self.ftype)
        bc[..., 0], bc[..., 1], bc[..., 2] = np.mgrid[
                                             box[0]:box[1]:complex(0, nx),
                                             box[2]:box[3]:complex(0, ny),
                                             box[4]:box[5]:complex(0, nz + 1)]
        return bc 

    ## @ingroup FDMInterface
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
        elif etype in {'cell', 3}:
            uh = np.zeros((nx+2*ex, ny+2*ex, nz+2*ex), dtype=dtype)
        else:
            raise ValueError(f'the entity `{entity}` is not correct!') 

        return uh

    ## @ingroup FDMInterface
    def gradient(self, f, order=1):
        """
        @brief 求网格函数 f 的梯度
        """
        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]
        fx, fy, fz= np.gradient(f, hx, hy, hz, edge_order=order)
        return fx, fy, fz
        
    ## @ingroup FDMInterface
    def divergence(self, f_x, f_y, f_z, order=1):
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

    ## @ingroup FDMInterface
    def laplace(self, f, order=1):
        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]
        fx, fy, fz = np.gradient(f, hx, hy, hz, edge_order=order)
        fxx, fxy, fxz = np.gradient(fx, hx, hy, hz, edge_order=order)
        fyx, fyy, fyz = np.gradient(fy, hx, hy, hz, edge_order=order)
        fzx, fzy, fzz = np.gradient(fz, hx, hy ,hz, edge_order=order)
        return fxx + fyy + fzz

    ## @ingroup FDMInterface
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
   
    ## @ingroup FDMInterface
    def interpolation(self, f, intertype='node'):
        """
        @brief
        """
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

    ## @ingroup FDMInterface
    def interpolate(self, f, intertype='node'):
        """
        """
        if intertype == 'node':
            node = self.node
            F = f(node)
        elif intertype in {'facex'}: # 法线和 x 轴平行的面
            xbc = self.facex_barycenter()
            F = f(xbc)
        elif intertype in {'facey'}: # 法线和 y 轴平行的面
            ybc = self.facey_barycenter()
            F = f(ybc)
        elif intertype in {'facez'}: # 法线和 z 轴平行的面
            zbc = self.facez_barycenter()
            F = f(zbc)
        elif intertype in {'face', 2}: # 所有的面
            xbc, ybc, zbc = self.face_barycenter()
            F = f(xbc), f(ybc), f(zbc)
        elif intertype in {'edgex'}: # 切向与 x 轴平行的边
            xbc = self.edgex_barycenter()
            F = f(xbc)
        elif intertype in {'edgey'}: # 切向与 y 轴平行的边
            ybc = self.edgey_barycenter()
            F = f(ybc)
        elif intertype in {'edgez'}: # 切向与 z 轴平行的边
            zbc = self.edgez_barycenter('edgez')
            F = f(zbc)
        elif intertype in {'edge', 1}: # 所有的边
            xbc, ybc, zbc = self.edge_barycenter()
            F = f(xbc), f(ybc), f(zbc)
        elif intertype in {'cell', 3}:
            bc = self.cell_barycenter()
            F = f(bc)
        return F
        pass

    ## @ingroup FDMInterface
    def error(self, h, nx, ny, nz, u, uh):
        """
        @brief 计算真解在网格点处与数值解的误差
        @param[in] u
        @param[in] uh
        """
        e = u - uh

        emax = np.max(np.abs(e))
        e0 = np.sqrt(h ** 2 * np.sum(e ** 2))

        el2 = np.sqrt(1 / ((nx - 1) * (ny - 1) * (nz - 1)) * np.sum(e ** 2))

        return emax, e0, el2

    ## @ingroup FDMInterface
    def elliptic_operator(self, d=3, c=None, r=None):                                                                                                                
        """
        @brief Assemble the finite difference matrix for a general elliptic operator.
        """
        pass

    ## @ingroup FDMInterface
    def laplace_operator(self):
        """
        @brief 构造笛卡尔网格上的 Laplace 离散算子，其中 x, y, z
        三个方向都是均匀剖分，但各自步长可以不一样
        @todo 处理带系数的情形
        """

        n0 = self.ds.nx + 1
        n1 = self.ds.ny + 1
        n2 = self.ds.nz + 1

        hx = self.h[0]
        hy = self.h[1]
        hz = self.h[2]

        cx = 1 / (hx ** 2)
        cy = 1 / (hy ** 2)
        cz = 1 / (hz ** 2)

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

    ## @ingroup FDMInterface
    def apply_dirichlet_bc(self, uh, A, f):
        """
        @brief
        """
        pass

    ## @ingroup FDMInterface
    def wave_equation(self, r, theta):
        """
        @brief
        """
        pass

    ## @ingroup FDMInterface                                                                                                                                         
    def fast_sweeping_method(self, phi0):
        """
        @brief 均匀网格上的 fast sweeping method
        @param[in] phi 是一个离散的水平集函数
        """
        pass

    ## @ingroup FEMInterface
    def geo_dimension(self):
        """
        @brief Get the geometry dimension of the mesh.
         
        @return The geometry dimension (3 for 3D mesh).
        """
        return 3

    ## @ingroup FEMInterface
    def top_dimension(self):
        """
        @brief Get the topological dimension of the mesh.
         
        @return The topological dimension (3 for 3D mesh).
        """
        return 3

    ## @ingroup FEMInterface
    def integrator(self, q, etype='cell'):
        pass

    ## @ingroup FEMInterface
    def bc_to_point(self, bc, index=np.s_[:]):
        pass

    ## @ingroup FEMInterface
    def entity(self, etype):
        """
        @brief Get the entity (either cell or node) based on the given entity type.

        @param[in] etype The type of entity, either 'cell', 3, 'face', 2, 'edge', 1, `node', 0.

        @return The cell, edeg, face, or node array based on the input entity type.

        @throws ValueError if the given etype is invalid.
        """
        pass

    ## @ingroup FEMInterface
    def entity_barycenter(self, etype='cell'):
        """
        @brief Get the entity (cell, face, edge, or  node) based on the given entity type.

        @param[in] etype The type of entity can be 'cell', 3, 'face', 2, 'edge',
        1, 'node', 0.

        @return The cell or node array based on the input entity type.

        @throws ValueError if the given etype is invalid

        @TODO 修改成一维和二维的样子
        """
        GD = self.geo_dimension()
        nx = self.ds.nx
        ny = self.ds.ny
        nz = self.ds.nz
        if etype in {'cell', 3}: # 所有单元
            return self.cell_barycenter().reshape(-1, 3) 
        elif etype in {'face', 2}: # 所有的面
            pass
        elif etype in {'edge', 1}: # 所有的边
            pass
        elif etype in {'node', 0}:
            return self.node.reshape(-1, 3)
        else:
            raise ValueError(f'the entity type `{etype}` is not correct!') 

    ## @ingroup FEMInterface
    def entity_measure(self, etype):
        """
        @brief
        """
        pass

    ## @ingroup FEMInterface
    def multi_index_matrix(self, p, etype=1):
        pass

    ## @ingroup FEMInterface
    def shape_function(self, bc, p=1):
        pass

    ## @ingroup FEMInterface
    def grad_shape_function(self, bc, p=1):
        pass

    ## @ingroup FEMInterface
    def number_of_local_ipoints(self, p, iptype='cell'):
        pass
    
    ## @ingroup FEMInterface
    def number_of_global_ipoints(self, p):
        pass

    ## @ingroup FEMInterface
    def interpolation_points(self, p):
        pass

    ## @ingroup FEMInterface
    def node_to_ipoint(self, p):
        pass

    ## @ingroup FEMInterface
    def edge_to_ipoint(self, p):
        pass

    ## @ingroup FEMInterface
    def face_to_ipoint(self, p):
        pass

    ## @ingroup FEMInterface
    def cell_to_ipoint(self, p):
        pass
