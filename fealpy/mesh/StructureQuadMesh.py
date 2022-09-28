import numpy as np

from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import triu, tril, diags, kron, eye
from .Mesh2d import Mesh2d

class StructureQuadMesh(Mesh2d):
    def __init__(self, box, nx, ny, itype=np.int_, ftype=np.float64):
        self.box = box
        self.ds = StructureQuadMeshDataStructure(nx, ny, itype)
        self.meshtype="quad"
        self.hx = (box[1] - box[0])/nx
        self.hy = (box[3] - box[2])/ny
        self.data = {}

        self.itype = itype
        self.ftype = ftype

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.meshdata = {}

    def uniform_refine(self, n=1, returnim=False):
        if returnim:
            nodeImatrix = []
        for i in range(n):
            nx = 2*self.ds.nx
            ny = 2*self.ds.ny
            itype = self.ds.itype
            self.ds = StructureQuadMeshDataStructure(nx, ny, itype)
            self.hx = (self.box[1] - self.box[0])/nx
            self.hy = (self.box[3] - self.box[2])/ny
            self.data = {}

            if returnim:
                A = self.interpolation_matrix()
                nodeImatrix.append(A)

        if returnim:
            return nodeImatrix

    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_Quad = 9
            return VTK_Quad
        elif etype in {'face', 'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE

    def to_vtk(self, etype='cell', fname=None):
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
        from .vtk_extent import vtk_cell_index, write_to_vtu
        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)
        
        cell = self.entity(etype)
        cellType = self.vtk_cell_type(etype)
        NV = cell.shape[-1]
        NC = len(cell)

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell]
        cell[:, 0] = NV

        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)

    def interpolation_matrix(self):
        """
        @brief  加密一次生成的矩阵
        """
        nx = self.ds.nx
        ny = self.ds.ny
        NNH = (nx//2+1)*(ny//2+1)
        NNh = self.number_of_nodes()

        I = np.arange(NNh).reshape(nx+1, -1)
        J = np.arange(NNH).reshape(nx//2+1, -1)

        ## (2i, 2j)
        I1 = I[::2, ::2].flat
        J1 = J.flat #(i,j)
        data = np.broadcast_to(1, (len(I1),))
        A = coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        ## (2i+1, 2j)
        I1 = I[1::2, ::2].flat
        J1 = J[:-1, :].flat #(i,j)
        data = np.broadcast_to(1/2, (len(I1),))
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[1:, :].flat #(i+1,j)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        ## (2i, 2j+1)
        I1 = I[::2, 1::2].flat
        J1 = J[:, :-1].flat #(i,j)
        data = np.broadcast_to(1/2, (len(I1),))
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        J1 = J[:, 1:].flat #(i,j+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))

        ## (2i+1, 2j+1)
        I1 = I[1::2, 1::2].flat
        J1 = J[:-1, :-1].flat #{i,j}
        data = np.broadcast_to(1/4, (len(I1),))
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))
        
        J1 = J[1:, :-1].flat # (i+1,j)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))
        
        J1 = J[:-1, 1:].flat # (i,j+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))
        
        J1 = J[1:, 1:].flat # (i+1,j+1)
        A += coo_matrix((data, (I1, J1)), shape=(NNh, NNH))


        return A

    def multi_index(self):
        NN = self.ds.NN
        nx = self.ds.nx
        ny = self.ds.ny
        i, j = np.mgrid[0:nx+1, 0:ny+1]
        index = np.zeros((NN, 2), dtype=self.itype)
        index[:, 0] = i.flat
        index[:, 1] = j.flat
        return index

    @property
    def node(self):
        NN = self.ds.NN
        nx = self.ds.nx
        ny = self.ds.ny
        box = self.box

        X, Y = np.mgrid[
                box[0]:box[1]:complex(0, nx+1),
                box[2]:box[3]:complex(0, ny+1)]
        node = np.zeros((NN, 2), dtype=self.ftype)
        node[:, 0] = X.flat
        node[:, 1] = Y.flat
        return node

    def number_of_nodes(self):
        return self.ds.NN

    def number_of_edges(self):
        return self.ds.NE

    def number_of_cells(self):
        return self.ds.NC

    def geo_dimension(self):
        return self.node.shape[1]

    def cell_area(self, index=None):
        NC = self.number_of_cells()
        node = self.entity('node')
        edge = self.entity('edge')
        edge2cell = self.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        v = self.edge_normal()
        val = np.sum(v*node[edge[:, 0], :], axis=1)

        a = np.zeros(NC, dtype=self.ftype)
        np.add.at(a, edge2cell[:, 0], val)
        np.add.at(a, edge2cell[isInEdge, 1], -val[isInEdge])
        a /=2
        return a

    def function(self, etype='node', dtype=None):
        """
        @brief 返回定义在节点、网格边、或者网格单元上离散函数（数组），元素取值为0
        """

        nx = self.ds.nx
        ny = self.ds.ny
        dtype = self.ftype if dtype is None else dtype

        if etype in {'node', 0}:
            uh = np.zeros((nx+1, ny+1), dtype=dtype)
        elif etype in {'edge', 1}:
            ex = np.zeros((nx, ny+1), dtype=dtype)
            ey = np.zeros((nx+1, ny), dtype=dtype)
            uh = (ex, ey)
        elif etype in {'edgex'}:
            uh = np.zeros((nx, ny+1), dtype=dtype)
        elif etype in {'edgey'}:
            uh = np.zeros((nx+1, ny), dtype=dtype)
        elif etype in {'cell', 2}:
            uh = np.zeros((nx, ny), dtype=dtype)
        return uh

    def data_edge_to_node(self, Ex, Ey):
        """
        @brief 
        """
        dx = self.function(etype='node') # (nx+1, ny+1)
        dy = self.function(etype='node') # (nx+1, ny+1)

        dx[0:-1, :] = Ex
        dx[-1, :] = Ex[-1, :]
        dx[1:-1, :] += Ex[1:, :]
        dx[1:-1, :] /=2.0

        dy[:, 0:-1] = Ey
        dy[:, -1] = Ey[:, -1]
        dy[:, 1:-1] += Ey[:, 1:]
        dy[:, 1:-1] /=2.0

        NN = len(dx.flat)
        data = np.zeros((NN, 2), dtype=Ex.dtype)
        data[:, 0] = dx.flat
        data[:, 1] = dy.flat

        return data


    def interpolation(self, f, intertype='node'):
        nx = self.ds.nx
        ny = self.ds.ny
        node = self.node
        if intertype == 'node':
            F = f(node)
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx+1, ny+1) + shape
            F = F.reshape(shape)
        elif intertype == 'edge':
            ec = self.entity_barycenter('edge')
            F = f(ec)
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]

            isXDEdge = self.ds.x_direction_edge_flag()
            shape = (nx, ny+1) + shape
            XF = F[isXDEdge].reshape(shape)

            isYDEdge = self.ds.y_direction_edge_flag()
            shape = (nx+1, ny) + shape
            YF = F[isYDEdge].reshape(shape)
            F = (XF, YF)

        elif intertype == 'edgex':
            isXDEdge = self.ds.x_direction_edge_flag()
            ec = self.entity_barycenter('edge')
            F = f(ec[isXDEdge])
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx, ny+1) + shape
            F = F.reshape(shape)
        elif intertype == 'edgey':
            isYDEdge = self.ds.y_direction_edge_flag()
            ec = self.entity_barycenter('edge')
            F = f(ec[isYDEdge])
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx+1, ny) + shape
            F = F.reshape(shape)
        elif intertype == 'cell':
            bc = self.entity_barycenter('cell')
            F = f(bc)
            shape = tuple() if len(F.shape) == 1 else F.shape[1:]
            shape = (nx, ny) + shape
            F = F.reshape(shape)
        return F

    def gradient(self, f):

        
        hx = self.hx
        hy = self.hy

        fx, fy = np.gradient(f, hx, hy)
        return fx, fy
        
    def div(self,fx,fy):
        hx = self.hx
        hy = self.hy
        fxx,fxy = np.gradient(fx,hx,hy)
        fyx,fyy = np.gradient(fy,hx,hy)
        f = fxx+fyy
        return f

    def laplace_operator(self):
        """
        @brief 构造笛卡尔网格上的 Laplace 离散算子，其中 x 方向和 y
        方向都均匀剖分，但步长可以不一样
        @todo 处理带系数的情形
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
        x = node[:, 0].reshape(self.ds.nx + 1, self.ds.ny + 1)
        y = node[:, 1].reshape(self.ds.nx + 1, self.ds.ny + 1)
        uh = uh.reshape(self.ds.nx + 1, self.ds.ny + 1)
        return axes.plot_surface(x, y, uh, cmap=cmap)


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


    def cell_location(self, px):
        """
        给定一组点，确定所有点所在的单元

        Parameter
        ---------
        px: numpy ndarray

        Note
        ----
        这里假设所有的点都在区域内部
        """
        box = self.box
        hx = self.hx
        hy = self.hy
        nx = self.ds.nx
        ny = self.ds.ny

        v = px - np.array(box[0::2], dtype=self.ftype)
        n0 = v[..., 0]//hx
        n1 = v[..., 1]//hy

        cidx = n0*ny + n1
        return cidx.astype(self.itype)

    def polation_interoperator(self, uh):
        """
        只适应在边界中点的插值
        """
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        isYDEdge = self.ds.y_direction_edge_flag()
        isXDEdge = self.ds.x_direction_edge_flag()
        isBDEdge = self.ds.boundary_edge_flag()
        bc = self.entity_barycenter()
        nx = self.ds.nx
        ny = self.ds.ny

        edge2cell = self.ds.edge_to_cell()
        cell2cell = self.ds.cell_to_cell()
        Pi = np.zeros(NE, dtype=self.ftype)

        I, = np.nonzero(~isBDEdge & isYDEdge)
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]

        Pi[I] = (uh[L] + uh[R])/2
                                       
        I, = np.nonzero(~isBDEdge & isXDEdge)
        L = edge2cell[I, 0]
        R = edge2cell[I, 1]
        Pi[I] = (uh[L] + uh[R])/2
        
        I, = np.nonzero(isBDEdge & isYDEdge)
        J = edge2cell[I, 0]
        L = cell2cell[J[ny:], 3]
        R = cell2cell[J[:ny], 1]
        
        Pi[I[:ny]] = (3*uh[J[:ny]] - uh[R])/2
        Pi[I[ny:]] = (3*uh[J[ny:]] - uh[L])/2

        I, = np.nonzero(isBDEdge & isXDEdge)
        J = edge2cell[I, 0]
        R = cell2cell[J[1::2], 0]
        L = cell2cell[J[::2], 2]
        Pi[I[1::2]] = (3*uh[J[1::2]] - uh[R])/2
        Pi[I[::2]] = (3*uh[J[::2]] - uh[L])/2
        
        return Pi


    def cbar_interpolation(self, ch, cellidx, xbar):
        nx = self.ds.nx
        ny = self.ds.ny
        cvalue = np.zeros(((ny+2), (nx+2)), dtype=self.ftype)
        cvalue[1:-1, 1:-1] = ch.reshape(ny, nx)

        Pi = self.polation_interoperator(ch)
        isYDEdge = self.ds.y_direction_edge_flag()
        isXDEdge = self.ds.x_direction_edge_flag()
        isBDEdge = self.ds.boundary_edge_flag()
        JY = isYDEdge & isBDEdge
        JX = isXDEdge & isBDEdge
        I, = np.nonzero(isYDEdge & isBDEdge)
        J, = np.nonzero(isXDEdge & isBDEdge)

        data = Pi[I]
        cvalue[::nx+1, 1:-1] = data.reshape(2, ny)
        data = Pi[J]
        cvalue[1:-1, ::ny+1] = data.reshape(nx, 2)

        cvalue[0, 0] = cvalue[0, 1] + cvalue[1, 0] - cvalue[1, 1]
        cvalue[-1, -1] = cvalue[-1, -2] + cvalue[-2, -1] - cvalue[-2, -2]
        cvalue[0, -1] = cvalue[0, -2] + cvalue[1, -1] - cvalue[1, -2]
        cvalue[-1, 0] = cvalue[-2, 0] + cvalue[-1, 1] - cvalue[-2, 1]
        
        NC = self.number_of_cells()
        newcellidx = np.zeros((NC, ), dtype=self.ftype)
        cell2cell = self.ds.cell_to_cell()
        isBDCell = self.ds.boundary_cell_flag()
        iscell = cellidx == cellidx

        for i in range(NC):
            for j in range(nx):
                if j*ny <= cellidx[i] < (j+1)*ny:
                    newcellidx[i] = cellidx[i] + 2*j + ny + 3


        wx1 = np.zeros((NC, ), dtype=self.ftype)
        wx2 = np.zeros((NC, ), dtype=self.ftype)
        wx3 = np.zeros((NC, ), dtype=self.ftype)
        wy1 = np.zeros((NC, ), dtype=self.ftype)
        wy2 = np.zeros((NC, ), dtype=self.ftype)
        wy3 = np.zeros((NC, ), dtype=self.ftype)


        bc = self.entity_barycenter('cell')
        ec = self.entity_barycenter('edge')
        incell = ~isBDCell & iscell
        Ic, = np.nonzero(incell)

        ## bc[Ic, 0] is (i, j)
        ## bc[cell2cell[Ic, 1], 0] is (i+1, j)
        ## bc[cell2cell[Ic, 3], 0] is (i-1, j)
        wx1[Ic] = (xbar[Ic, 0] - bc[Ic, 0])\
                *(xbar[Ic, 0] - bc[cell2cell[Ic, 1], 0])\
                /(bc[cell2cell[Ic, 3], 0] - bc[Ic, 0])\
                /(bc[cell2cell[Ic, 3], 0] - bc[cell2cell[Ic, 1], 0])
        wx2[Ic] = (xbar[Ic, 0] - bc[cell2cell[Ic, 3], 0])\
                *(xbar[Ic, 0] - bc[cell2cell[Ic, 1], 0])\
                /(bc[Ic, 0] - bc[cell2cell[Ic, 3], 0])\
                /(bc[Ic, 0] - bc[cell2cell[Ic, 1], 0])
        wx3[Ic] = (xbar[Ic, 0] - bc[cell2cell[Ic, 3], 0])\
                *(xbar[Ic, 0] - bc[Ic, 0])\
                /(bc[cell2cell[Ic, 1], 0] - bc[cell2cell[Ic, 3], 0])\
                /(bc[cell2cell[Ic, 1], 0] - bc[Ic, 0])

        ## bc[Ic, 1] is (i, j)
        ## bc[cell2cell[Ic, 0], 1] is (i, j-1)
        ## bc[cell2cell[Ic, 2], 1] is (i, j+1)

 
        wy1[Ic] = (xbar[Ic, 1] - bc[Ic, 1])\
                *(xbar[Ic, 1] - bc[cell2cell[Ic, 2], 1])\
                /(bc[cell2cell[Ic, 0], 1] - bc[Ic, 1])\
                /(bc[cell2cell[Ic, 0], 1] - bc[cell2cell[Ic, 2], 1])
        wy2[Ic] = (xbar[Ic, 1] - bc[cell2cell[Ic, 0], 1])\
                *(xbar[Ic, 1] - bc[cell2cell[Ic, 2], 1])\
                /(bc[Ic, 1] - bc[cell2cell[Ic, 0], 1])\
                /(bc[Ic, 1] - bc[cell2cell[Ic, 2], 1])
        wy3[Ic] = (xbar[Ic, 1] - bc[cell2cell[Ic, 0], 1])\
                *(xbar[Ic, 1] - bc[Ic, 1])\
                /(bc[cell2cell[Ic, 2], 1] - bc[cell2cell[Ic, 0], 1])\
                /(bc[cell2cell[Ic, 2], 1] - bc[Ic, 1])
 

        cell2edge = self.ds.cell_to_edge()
        edge2cell = self.ds.edge_to_cell()

        LRCell = edge2cell[I, 0]
        UACell = edge2cell[J, 0]

        LC = LRCell[:ny]
        P1 = ec[I[:ny], 0]
        ## bc[LC, 0] is (i, j)
        ## bc[cell2cell[LC, 1], 0] is (i+1, j)
        ## P1 is (i-1, j)
 
        wx1[LC] = (xbar[LC, 0] - bc[LC, 0])\
                *(xbar[LC, 0] - bc[cell2cell[LC, 1], 0])\
                /(P1 - bc[LC, 0])\
                /(P1 - bc[cell2cell[LC, 1], 0])
        wx2[LC] = (xbar[LC, 0] - P1)\
                *(xbar[LC, 0] - bc[cell2cell[LC, 1], 0])\
                /(bc[LC, 0] - P1)\
                /(bc[LC, 0] - bc[cell2cell[LC, 1], 0])
        wx3[LC] = (xbar[LC, 0] - P1)\
                *(xbar[LC, 0] - bc[LC, 0])\
                /(bc[cell2cell[LC, 1], 0] - P1)\
                /(bc[cell2cell[LC, 1], 0] - bc[LC, 0])

        wy1[LC[1:-1]] = (xbar[LC[1:-1], 1] - bc[LC[1:-1], 1])\
                *(xbar[LC[1:-1], 1] - bc[cell2cell[LC[1:-1], 2], 1])\
                /(bc[cell2cell[LC[1:-1], 0], 1] - bc[LC[1:-1], 1])\
                /(bc[cell2cell[LC[1:-1], 0], 1] - bc[cell2cell[LC[1:-1], 2], 1])
        wy2[LC[1:-1]] = (xbar[LC[1:-1], 1] - bc[cell2cell[LC[1:-1], 0], 1])\
                *(xbar[LC[1:-1], 1] - bc[cell2cell[LC[1:-1], 2], 1])\
                /(bc[LC[1:-1], 1] - bc[cell2cell[LC[1:-1], 0], 1])\
                /(bc[LC[1:-1], 1] - bc[cell2cell[LC[1:-1], 2], 1])
        wy3[LC[1:-1]] = (xbar[LC[1:-1], 1] - bc[cell2cell[LC[1:-1], 0], 1])\
                *(xbar[LC[1:-1], 1] - bc[LC[1:-1], 1])\
                /(bc[cell2cell[LC[1:-1], 2], 1] - bc[cell2cell[LC[1:-1], 0], 1])\
                /(bc[cell2cell[LC[1:-1], 2], 1] - bc[LC[1:-1], 1])
 

        RC = LRCell[ny:]
        P1 = ec[I[ny:], 0]
        ## bc[RC, 0] is (i, j)
        ## bc[cell2cell[RC, 3], 0] is (i-1, j)
        ## P1 is (i+1, j)
 
        wx1[RC] = (xbar[RC, 0] - bc[RC, 0])\
                *(xbar[RC, 0] - P1)\
                /(bc[cell2cell[RC, 3], 0] - bc[RC, 0])\
                /(bc[cell2cell[RC, 3], 0] - P1)
        wx2[RC] = (xbar[RC, 0] - bc[cell2cell[RC, 3], 0])\
                *(xbar[RC, 0] - P1)\
                /(bc[RC, 0] - bc[cell2cell[RC, 3], 0])\
                /(bc[RC, 0] - P1)
        wx3[RC] = (xbar[RC, 0] - bc[cell2cell[RC, 3], 0])\
                *(xbar[RC, 0] - bc[RC, 0])\
                /(P1 - bc[cell2cell[RC, 3], 0])\
                /(P1 - bc[RC, 0])

        wy1[RC[1:-1]] = (xbar[RC[1:-1], 1] - bc[RC[1:-1], 1])\
                *(xbar[RC[1:-1], 1] - bc[cell2cell[RC[1:-1], 2], 1])\
                /(bc[cell2cell[RC[1:-1], 0], 1] - bc[RC[1:-1], 1])\
                /(bc[cell2cell[RC[1:-1], 0], 1] - bc[cell2cell[RC[1:-1], 2], 1])
        wy2[RC[1:-1]] = (xbar[RC[1:-1], 1] - bc[cell2cell[RC[1:-1], 0], 1])\
                *(xbar[RC[1:-1], 1] - bc[cell2cell[RC[1:-1], 2], 1])\
                /(bc[RC[1:-1], 1] - bc[cell2cell[RC[1:-1], 0], 1])\
                /(bc[RC[1:-1], 1] - bc[cell2cell[RC[1:-1], 2], 1])
        wy3[RC[1:-1]] = (xbar[RC[1:-1], 1] - bc[cell2cell[RC[1:-1], 0], 1])\
                *(xbar[RC[1:-1], 1] - bc[RC[1:-1], 1])\
                /(bc[cell2cell[RC[1:-1], 2], 1] - bc[cell2cell[RC[1:-1], 0], 1])\
                /(bc[cell2cell[RC[1:-1], 2], 1] - bc[RC[1:-1], 1])
 
        UC = UACell[::2]
        P1 = ec[J[::2], 1]
        ## bc[UC, 1] is (i, j)
        ## bc[cell2cell[UC, 2], 1] is (i, j+1)
        ## P1 is (i, j-1)
        wx1[UC[1:-1]] = (xbar[UC[1:-1], 0] - bc[UC[1:-1], 0])\
                *(xbar[UC[1:-1], 0] - bc[cell2cell[UC[1:-1], 1], 0])\
                /(bc[cell2cell[UC[1:-1], 3], 0] - bc[UC[1:-1], 0])\
                /(bc[cell2cell[UC[1:-1], 3], 0] - bc[cell2cell[UC[1:-1], 1], 0])
        wx2[UC[1:-1]] = (xbar[UC[1:-1], 0] - bc[cell2cell[UC[1:-1], 3], 0])\
                *(xbar[UC[1:-1], 0] - bc[cell2cell[UC[1:-1], 1], 0])\
                /(bc[UC[1:-1], 0] - bc[cell2cell[UC[1:-1], 3], 0])\
                /(bc[UC[1:-1], 0] - bc[cell2cell[UC[1:-1], 1], 0])
        wx3[UC[1:-1]] = (xbar[UC[1:-1], 0] - bc[cell2cell[UC[1:-1], 3], 0])\
                *(xbar[UC[1:-1], 0] - bc[UC[1:-1], 0])\
                /(bc[cell2cell[UC[1:-1], 1], 0] - bc[cell2cell[UC[1:-1], 3], 0])\
                /(bc[cell2cell[UC[1:-1], 1], 0] - bc[UC[1:-1], 0])

        wy1[UC] = (xbar[UC, 1] - bc[UC, 1])\
                *(xbar[UC, 1] - bc[cell2cell[UC, 2], 1])\
                /(P1 - bc[UC, 1])\
                /(P1 - bc[cell2cell[UC, 2], 1])
        wy2[UC] = (xbar[UC, 1] - P1)\
                *(xbar[UC, 1] - bc[cell2cell[UC, 2], 1])\
                /(bc[UC, 1] - P1)\
                /(bc[UC, 1] - bc[cell2cell[UC, 2], 1])
        wy3[UC] = (xbar[UC, 1] - P1)\
                *(xbar[UC, 1] - bc[UC, 1])\
                /(bc[cell2cell[UC, 2], 1] - P1)\
                /(bc[cell2cell[UC, 2], 1] - bc[UC, 1])
 
        AC = UACell[1::2]
        P1 = ec[J[1::2], 1]


        ## bc[AC, 1] is (i, j)
        ## bc[cell2cell[AC, 0], 1] is (i, j-1)
        ## P1 is (i, j+1)

        wx1[AC[1:-1]] = (xbar[AC[1:-1], 0] - bc[AC[1:-1], 0])\
                *(xbar[AC[1:-1], 0] - bc[cell2cell[AC[1:-1], 1], 0])\
                /(bc[cell2cell[AC[1:-1], 3], 0] - bc[AC[1:-1], 0])\
                /(bc[cell2cell[AC[1:-1], 3], 0] - bc[cell2cell[AC[1:-1], 1], 0])
        wx2[AC[1:-1]] = (xbar[AC[1:-1], 0] - bc[cell2cell[AC[1:-1], 3], 0])\
                *(xbar[AC[1:-1], 0] - bc[cell2cell[AC[1:-1], 1], 0])\
                /(bc[AC[1:-1], 0] - bc[cell2cell[AC[1:-1], 3], 0])\
                /(bc[AC[1:-1], 0] - bc[cell2cell[AC[1:-1], 1], 0])
        wx3[AC[1:-1]] = (xbar[AC[1:-1], 0] - bc[cell2cell[AC[1:-1], 3], 0])\
                *(xbar[AC[1:-1], 0] - bc[AC[1:-1], 0])\
                /(bc[cell2cell[AC[1:-1], 1], 0] - bc[cell2cell[AC[1:-1], 3], 0])\
                /(bc[cell2cell[AC[1:-1], 1], 0] - bc[AC[1:-1], 0])


        wy1[AC] = (xbar[AC, 1] - bc[AC, 1])\
                *(xbar[AC, 1] - P1)\
                /(bc[cell2cell[AC, 0], 1] - bc[AC, 1])\
                /(bc[cell2cell[AC, 0], 1] - P1)
        wy2[AC] = (xbar[AC, 1] - bc[cell2cell[AC, 0], 1])\
                *(xbar[AC, 1] - P1)\
                /(bc[AC, 1] - bc[cell2cell[AC, 0], 1])\
                /(bc[AC, 1] - P1)
        wy3[AC] = (xbar[AC, 1] - bc[cell2cell[AC, 0], 1])\
                *(xbar[AC, 1] - bc[AC, 1])\
                /(P1 - bc[cell2cell[AC, 0], 1])\
                /(P1 - bc[AC, 1])
        
        i = np.zeros((NC,), dtype=self.itype)
        j = np.zeros((NC,), dtype=self.itype)
        i[:] = newcellidx // (ny+2)
        j[:] = newcellidx % (ny+2)


        cbar = wy1[cellidx]*(wx1[cellidx]*cvalue[i-1, j-1] \
             + wx2[cellidx]*cvalue[i, j-1] + wx3[cellidx]*cvalue[i+1, j-1])\
             + wy2[cellidx]*(wx1[cellidx]*cvalue[i-1, j] \
             + wx2[cellidx]*cvalue[i, j] + wx3[cellidx]*cvalue[i+1, j])\
             + wy3[cellidx]*(wx1[cellidx]*cvalue[i-1, j+1] \
             + wx2[cellidx]*cvalue[i, j+1] + wx3[cellidx]*cvalue[i+1, j+1])\


        return cbar


class StructureQuadMeshDataStructure:
    localEdge = np.array([(0,1),(1,2),(2,3),(3,0)])
    ccw = np.array([0, 1, 2, 3])
    V = 4
    E = 4
    F = 1
    def __init__(self, nx, ny, itype):
        self.nx = nx
        self.ny = ny
        self.NN = (nx+1)*(ny+1)
        self.NE = ny*(nx+1) + nx*(ny+1)
        self.NC = nx*ny
        self.itype = itype
 
    def number_of_nodes_of_cells(self):
        return self.V

    def number_of_edges_of_cells(self):
        return self.E

    def number_of_faces_of_cells(self):
        return self.E

    def number_of_vertices_of_cells(self):
        return self.V

    @property
    def cell(self):

        nx = self.nx
        ny = self.ny

        NN = self.NN
        NC = self.NC
        cell = np.zeros((NC, 4), dtype=self.itype)
        idx = np.arange(NN).reshape(nx+1, ny+1)
        c = idx[:-1, :-1]
        cell[:, 0] = c.flat
        cell[:, 1] = cell[:, 0] + ny + 1
        cell[:, 2] = cell[:, 1] + 1
        cell[:, 3] = cell[:, 0] + 1
        return cell

    @property
    def edge(self):
        """
        @brief 生成网格中所有的边
        @todo 把顺序换为先 x 方向的边，后 y 方向的边。
        """
        nx = self.nx
        ny = self.ny

        NN = self.NN
        NE = self.NE

        idx = np.arange(NN, dtype=self.itype).reshape(nx+1, ny+1)
        edge = np.zeros((NE, 2), dtype=self.itype)

        NE0 = 0
        NE1 = ny*(nx+1)
        edge[NE0:NE1, 0] = idx[:, :-1].flat
        edge[NE0:NE1, 1] = idx[:, 1:].flat
        edge[NE0:NE0+ny, :] = edge[NE0:NE0+ny, -1::-1]

        NE0 = NE1
        NE1 += nx*(ny+1)
        edge[NE0:NE1, 0] = idx[:-1, :].flat
        edge[NE0:NE1, 1] = idx[1:, :].flat
        edge[NE1:NE0:-nx-1, :] = edge[NE1:NE0:-nx-1, -1::-1]
        return edge

    @property
    def edge2cell(self):

        nx = self.nx
        ny = self.ny

        NC = self.NC
        NE = self.NE

        edge2cell = np.zeros((NE, 4), dtype=self.itype)

        idx = np.arange(NC).reshape(nx, ny).T

        # y direction
        idx0 = np.arange((nx+1)*ny, dtype=self.itype).reshape(nx+1, ny).T
        #left element
        edge2cell[idx0[:,1:], 0] = idx
        edge2cell[idx0[:,1:], 2] = 1
        edge2cell[idx0[:,0], 0] = idx[:,0]
        edge2cell[idx0[:,0], 2] = 3
        

        #right element
        edge2cell[idx0[:,:-1], 1] = idx
        edge2cell[idx0[:,:-1], 3] = 3
        edge2cell[idx0[:,-1], 1] = idx[:,-1]
        edge2cell[idx0[:,-1], 3] = 1

        # x direction 
        idx1 = np.arange(nx*(ny+1),dtype=self.itype).reshape(nx, ny+1).T
        NE0 = ny*(nx+1)
        #left element
        edge2cell[NE0+idx1[:-1], 0] = idx
        edge2cell[NE0+idx1[:-1], 2] = 0
        edge2cell[NE0+idx1[-1], 0] = idx[-1]
        edge2cell[NE0+idx1[-1], 2] = 2

        #right element
        edge2cell[NE0+idx1[1:], 1] = idx
        edge2cell[NE0+idx1[1:], 3] = 2
        edge2cell[NE0+idx1[0],1] = idx[0]
        edge2cell[NE0+idx1[0], 3] = 0

        return edge2cell

    def cell_to_node(self):
        """ 
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = np.repeat(range(NC), V)
        val = np.ones(self.V*NC, dtype=np.bool)
        cell2node = csr_matrix((val, (I, cell.flatten())), shape=(NC, NN), dtype=np.bool)
        return cell2node

    def cell_to_edge(self, sparse=False):
        """ The neighbor information of cell to edge
        """
        NE = self.NE
        NC = self.NC
        E = self.E

        edge2cell = self.edge2cell

        if sparse == False:
            cell2edge = np.zeros((NC, E), dtype=self.itype)
            cell2edge[edge2cell[:, 0], edge2cell[:, 2]] = np.arange(NE,
                    dtype=self.itype)
            cell2edge[edge2cell[:, 1], edge2cell[:, 3]] = np.arange(NE,
                    dtype=self.itype)
            return cell2edge
        else:
            val = np.ones(2*NE, dtype=np.bool)
            I = edge2cell[:, [0, 1]].flatten()
            J = np.repeat(range(NE), 2)
            cell2edge = csr_matrix(
                    (val, (I, J)), 
                    shape=(NC, NE), dtype=np.bool)
            return cell2edge 

    def cell_to_edge_sign(self, sparse=False):
        NC = self.NC
        E = self.E

        edge2cell = self.edge2cell
        if sparse == False:
            cell2edgeSign = np.zeros((NC, E), dtype=np.bool)
            cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = True
        else:
            val = np.ones(NE, dtype=np.bool)
            cell2edgeSign = csr_matrix(
                    (val, (edge2cell[:, 0], range(NE))),
                    shape=(NC, NE), dtype=np.bool)
        return cell2edgeSign

    def cell_to_cell(self, return_sparse=False, return_boundary=True, return_array=False):
        """ Consctruct the neighbor information of cells
        """
        if return_array:                                                             
             return_sparse = False
             return_boundary = False
 
        NC = self.NC
        E = self.E
        edge2cell = self.edge2cell
        if (return_sparse == False) & (return_array == False):
            E = self.E
            cell2cell = np.zeros((NC, E), dtype=np.int)
            cell2cell[edge2cell[:, 0], edge2cell[:, 2]] = edge2cell[:, 1]
            cell2cell[edge2cell[:, 1], edge2cell[:, 3]] = edge2cell[:, 0]
            return cell2cell
        NE = self.NE
        val = np.ones((NE,), dtype=np.bool)
        if return_boundary:
            cell2cell = coo_matrix(
                    (val, (edge2cell[:, 0], edge2cell[:, 1])),
                    shape=(NC, NC), dtype=np.bool)
            cell2cell += coo_matrix(
                    (val, (edge2cell[:, 1], edge2cell[:, 0])),
                    shape=(NC, NC), dtype=np.bool)
            return cell2cell.tocsr()
        else:
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            cell2cell = coo_matrix(
                    (val[isInEdge], (edge2cell[isInEdge, 0], edge2cell[isInEdge, 1])),
                    shape=(NC, NC), dtype=np.bool)
            cell2cell += coo_matrix(
                    (val[isInEdge], (edge2cell[isInEdge, 1], edge2cell[isInEdge, 0])),
                    shape=(NC, NC), dtype=np.bool)
            cell2cell = cell2cell.tocsr()
            if return_array == False:
                return cell2cell
            else:
                nn = cell2cell.sum(axis=1).reshape(-1)
                _, adj = cell2cell.nonzero()
                adjLocation = np.zeros(NC+1, dtype=np.int32)
                adjLocation[1:] = np.cumsum(nn)
                return adj.astype(np.int32), adjLocation

    def edge_to_node(self, sparse=False):
        NN = self.NN
        NE = self.NE

        edge = self.edge
        if sparse == False:
            return edge
        else:
            edge = self.edge
            I = np.repeat(range(NE), 2)
            J = edge.flat
            val = np.ones(2*NE, dtype=np.bool)
            edge2node = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool)
            return edge2node

    def edge_to_edge(self, sparse=False):
        edge2node = self.edge_to_node()
        return edge2node*edge2node.tranpose()

    def edge_to_cell(self, sparse=False):
        if sparse==False:
            return self.edge2cell
        else:
            NC = self.NC
            NE = self.NE
            I = np.repeat(range(NF), 2)
            J = self.edge2cell[:, [0, 1]].flatten()
            val = np.ones(2*NE, dtype=np.bool)
            face2cell = csr_matrix((val, (I, J)), shape=(NE, NC), dtype=np.bool)
            return face2cell 

    def node_to_node(self):
        """ The neighbor information of nodes
        """
        NN = self.NN
        NE = self.NE
        edge = self.edge
        I = edge.flat
        J = edge[:,[1,0]].flat
        val = np.ones((2*NE,), dtype=np.bool)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN),dtype=np.bool)
        return node2node

    def node_to_edge(self):
        NN = self.NN
        NE = self.NE
        
        edge = self.edge
        I = edge.flat
        J = np.repeat(range(NE), 2)
        val = np.ones(2*NE, dtype=np.bool)
        node2edge = csr_matrix((val, (I, J)), shape=(NE, NN), dtype=np.bool)
        return node2edge

    def node_to_cell(self, localidx=False):
        """
        """
        NN = self.NN
        NC = self.NC
        V = self.V

        cell = self.cell

        I = cell.flat 
        J = np.repeat(range(NC), V)

        if localidx == True:
            val = ranges(V*np.ones(NC, dtype=np.int), start=1) 
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.int)
        else:
            val = np.ones(V*NC, dtype=np.bool)
            node2cell = csr_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool)
        return node2cell


    def boundary_node_flag(self):
        NN = self.NN
        edge = self.edge
        isBdEdge = self.boundary_edge_flag()
        isBdPoint = np.zeros((NN,), dtype=np.bool)
        isBdPoint[edge[isBdEdge,:]] = True
        return isBdPoint

    def boundary_edge_flag(self):
        edge2cell = self.edge2cell
        return edge2cell[:, 0] == edge2cell[:, 1]

    def boundary_cell_flag(self, bctype=None):
        """
        Parameters
        ----------
        bctype : None or 0, 1, 2 ,3
        """
        NC = self.NC

        if bctype is None:
            edge2cell = self.edge2cell
            isBdCell = np.zeros((NC,), dtype=np.bool)
            isBdEdge = self.boundary_edge_flag()
            isBdCell[edge2cell[isBdEdge,0]] = True

        else:
            cell2cell = self.cell_to_cell()
            isBdCell = cell2cell[:, bctype] == np.arange(NC)
        return isBdCell 

    def boundary_node_index(self):
        isBdPoint = self.boundary_node_flag()
        idx, = np.nonzero(isBdPoint)
        return idx 

    def boundary_edge_index(self):
        isBdEdge = self.boundary_edge_flag()
        idx, = np.nonzero(isBdEdge)
        return idx 

    def boundary_cell_index(self, bctype=None):
        isBdCell = self.boundary_cell_flag(bctype)
        idx, = np.nonzero(isBdCell)
        return idx 

    def y_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        return np.arange(ny*(nx+1))

    def x_direction_edge_index(self):
        nx = self.nx
        ny = self.ny
        NE = self.NE
        return np.arange(ny*(nx+1), NE)

    def y_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        NE = self.NE
        isYDEdge = np.zeros(NE, dtype=np.bool)
        isYDEdge[:ny*(nx+1)] = True
        return isYDEdge 

    def x_direction_edge_flag(self):
        nx = self.nx
        ny = self.ny
        NE = self.NE
        isXDEdge = np.zeros(NE, dtype=np.bool)
        isXDEdge[ny*(nx+1):] = True
        return isXDEdge  

    def left_boundary_node_index(self):
        nx = self.nx
        ny = self.ny
        return np.arange(ny+1)

    def right_boundary_node_index(self):
        nx = self.nx
        ny = self.ny
        NN = self.NN
        return np.arange(NN-ny-1, NN)

    def bottom_boundary_node__index(self):
        nx = self.nx
        ny = self.ny
        NN = self.NN 
        return np.arange(0, NN-ny, ny+1)

    def up_boundary_node_index(self):
        nx = self.nx
        ny = self.ny
        NN = self.NN
        return np.arange(ny, NN, ny+1)

    def peoriod_matrix(self):
        """
        we can get a matarix under periodic boundary condition 
        """
        nx = self.nx
        ny = self.ny
        NN = self.NN
        isPNode = np.zeros(NN, dtype=np.bool)
        lidx = self.left_boundary_node_index()
        ridx = self.right_boundary_node_index()
        bidx = self.bottom_boundary_node__index()
        uidx = self.up_boundary_node_index()

        isPNode[ridx] = True
        isPNode[uidx] = True
        NC = nx*ny
        #First, we get the inner elements , the left boundary and the lower boundary of the matrix.
        val = np.ones(NC, dtype = np.bool)
        I = np.arange(NN)[~isPNode]
        J = range(NC) 
        C = coo_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool)
        #second,  we make the upper boundary equal to the lower boundary.
        val = np.ones(nx, dtype=np.bool) 
        I = np.arange(NN)[uidx[:-1]]
        J = np.arange(0, NC-ny+1, ny)
        C += coo_matrix((val, (I, J)), shape=(NN, NC), dtype=np.bool)
        #thrid, we make the right boundary equal to the left boundary.
        val = np.ones(ny+1, dtype=np.bool)
        I = np.arange(NN)[ridx]
        J = np.arange(ny+1)
        J[-1] = 0
        C += coo_matrix((val,(I, J)), shape=(NN, NC), dtype=np.bool)

        return C

