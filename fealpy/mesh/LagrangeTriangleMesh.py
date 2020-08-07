import numpy as np
from ..quadrature import TriangleQuadrature, GaussLegendreQuadrature

from .Mesh2d import Mesh2d, Mesh2dDataStructure

from .TriangleMesh import TriangleMesh

from .multi_index import multi_index_matrix
from .multi_index import lagrange_shape_function
from .multi_index import lagrange_grad_shape_function

class LagrangeTriangleMesh(Mesh2d):
    def __init__(self, node, cell, p=1, surface=None):

        mesh = TriangleMesh(node, cell) 
        dof = LagrangeTriangleDof2d(mesh, p)

        self.p = p
        self.node = dof.interpolation_points()

        if surface is not None:
            self.node, _ = surface.project(self.node)
   
        self.ds = LagrangeTriangleMeshDataStructure(dof)

        self.GD = node.shape[1]
        self.TD = 2

        self.meshtype = 'ltri'
        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.multi_index_matrix = multi_index_matrix

    def number_of_corner_nodes(self):
        return self.ds.NCN


    def lagrange_dof(self, p):
        return LagrangeTriangleDof2d(self, p)

    def vtk_cell_type(self, etype='cell'):
        """

        Notes
        -----
            返回网格单元对应的 vtk类型。
        """
        if etype in {'cell', 2}:
            VTK_LAGRANGE_TRIANGLE = 69
            return VTK_LAGRANGE_TRIANGLE 
        elif etype in {'face', 'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE

    def to_vtk(self, etype='cell', index=np.s_[:], fname=None):
        """
        Parameters
        ----------

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        from .vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        index = vtk_cell_index(self.p, cellType)
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell[:, index]]
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)

    def integrator(self, q, etype='cell'):
        if etype in {'cell', 2}:
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            return GaussLegendreQuadrature(q)

    def cell_area(self, q=None):
        """

        Notes
        -----
            
        """
        p = self.p
        q = p if q is None else q

        qf = self.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs)
        n = np.cross(J[..., 0], J[..., 1], axis=-1)
        l = np.sqrt(np.sum(n**2, axis=-1))
        a = np.einsum('i, ij->j', ws, l)/2.0
        return a

    def edge_length(self, q=None):
        """

        """
        p = self.p
        q = p if q is None else q

        qf = self.integrator(q, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights() 

        J = self.jacobi_matrix(bcs)
        l = np.sqrt(np.sum(J**2, axis=-1))
        a = np.einsum('i, ij->j', ws, l)/2.0
        return a

    def bc_to_point(self, bc, etype='cell', index=np.s_[:]):
        node = self.node
        entity = self.entity(etype)[index] # 
        phi = self.lagrange_basis(bc, etype=etype) # (NQ, 1, ldof)
        p = np.einsum('ijk, jkn->ijn', phi, node[entity])
        return p

    def jacobi_matrix(self, bc, index=np.s_[:], etype='cell'):
        entity = self.entity(etype)
        grad = self.lagrange_grad_basis(bc, index=index, etype=etype)

        # the tranpose of the jacobi matrix between S_p and S_h
        J = np.einsum(
                'ijn, ...ijk->...ink',
                self.node[entity[index], :], grad)
        return J

    def shape_function(self, bc, p=None):
        p = self.p if p is None else p
        return lagrange_shape_function(bc, p)


    def grad_shape_function(self, bc, p=None):
        """

        Notes
        -----
        计算单元形函数关于参考单元变量的梯度。

        lambda_0 = 1 - xi - eta
        lambda_1 = xi
        lambda_2 = eta
        """
        p = self.p if p is None else p 
        TD = bc.shape[-1] - 1
        if TD == 2:
            Dlambda = np.array([[-1, -1], [1, 0], [0, 1]], dtype=self.ftype)
        else:
            Dlambda = np.array([[-1], [1]], dtype=self.ftype)
        R = lagrange_grad_shape_function(bc, p) # (..., ldof, TD+1)

        gphi = np.einsum('...ij, jn->...in', R, Dlambda) # (NQ, ldof, TD)
        return gphi[..., None, :, :] #(..., 1, ldof, TD)

    def print(self):

        """

        Notes
        -----
            打印网格信息， 用于调试。
        """

        node = self.entity('node')
        print('node:')
        for i, v in enumerate(node):
            print(i, ": ", v)

        edge = self.entity('edge')
        print('edge:')
        for i, e in enumerate(edge):
            print(i, ": ", e)

        cell = self.entity('cell')
        print('cell:')
        for i, c in enumerate(cell):
            print(i, ": ", c)

        edge2cell = self.ds.edge_to_cell()
        print('edge2cell:')
        for i, ec in enumerate(edge2cell):
            print(i, ": ", ec)

class LagrangeTriangleMeshDataStructure(Mesh2dDataStructure):
    def __init__(self, dof):
        self.NCN = dof.mesh.number_of_nodes()
        self.cell = dof.cell_to_dof()
        self.edge = dof.edge_to_dof()
        self.edge2cell = dof.mesh.ds.edge_to_cell()

        self.NN = dof.number_of_global_dofs() 
        self.NE = len(self.edge)
        self.NC = len(self.cell)

        self.V = dof.number_of_local_dofs() 
        self.E = 3

class LagrangeTriangleDof2d():
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p
        self.multiIndex = multi_index_matrix[2](p)

    def is_on_node_local_dof(self):
        p = self.p
        isNodeDof = np.sum(self.multiIndex == p, axis=-1) > 0
        return isNodeDof

    def is_on_edge_local_dof(self):
        return self.multiIndex == 0

    def is_boundary_dof(self, threshold=None):

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        edge2dof = self.edge_to_dof()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        isBdDof[edge2dof[index]] = True
        return isBdDof

    def face_to_dof(self):
        return self.edge_to_dof()

    @property
    def edge2dof(self):
        return self.edge_to_dof()

    def edge_to_dof(self):
        """

        TODO
        ----
        1. 只取一部分边上的自由度
        """
        p = self.p
        mesh = self.mesh
        edge = mesh.entity('edge')

        if p == mesh.p:
            return edge

        edge2dof = np.zeros((NE, p+1), dtype=np.int)
        edge2dof[:, [0, -1]] = edge[:, [0, -1]] # edge 可以是高次曲线
        if p > 1:
            NN = mesh.number_of_corner_nodes() # 注意这里只是单元角点的个数
            NE = mesh.number_of_edges()
            edge2dof[:, 1:-1] = NCN + np.arange(NE*(p-1)).reshape(NE, p-1)

        return edge2dof

    @property
    def cell2dof(self):
        """
        
        Notes
        -----
            把这个方法属性化，保证程序接口兼容性
        """
        return self.cell_to_dof()


    def cell_to_dof(self):
        """

        TODO
        ----
        1. 只取一部分单元上的自由度
        """

        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell') # cell 可以是高次单元

        if p == mesh.p:
            return cell 

        # 空间自由度和网格的自由度不一致时，重新构造单元自由度矩阵
        NN = mesh.number_of_corner_nodes() # 注意这里只是单元角点的个数
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        cell2dof = np.zeros((NC, ldof), dtype=np.int_)

        isEdgeDof = self.is_on_edge_local_dof()
        edge2dof = self.edge_to_dof()
        cell2edgeSign = mesh.ds.cell_to_edge_sign()
        cell2edge = mesh.ds.cell_to_edge()

        cell2dof[np.ix_(cell2edgeSign[:, 0], isEdgeDof[:, 0])] = \
                edge2dof[cell2edge[cell2edgeSign[:, 0], [0]], :]
        cell2dof[np.ix_(~cell2edgeSign[:, 0], isEdgeDof[:,0])] = \
                edge2dof[cell2edge[~cell2edgeSign[:, 0], [0]], -1::-1]

        cell2dof[np.ix_(cell2edgeSign[:, 1], isEdgeDof[:, 1])] = \
                edge2dof[cell2edge[cell2edgeSign[:, 1], [1]], -1::-1]
        cell2dof[np.ix_(~cell2edgeSign[:, 1], isEdgeDof[:,1])] = \
                edge2dof[cell2edge[~cell2edgeSign[:, 1], [1]], :]

        cell2dof[np.ix_(cell2edgeSign[:, 2], isEdgeDof[:, 2])] = \
                edge2dof[cell2edge[cell2edgeSign[:, 2], [2]], :]
        cell2dof[np.ix_(~cell2edgeSign[:, 2], isEdgeDof[:,2])] = \
                edge2dof[cell2edge[~cell2edgeSign[:, 2], [2]], -1::-1]
        if p > 2:
            base = NN + (p-1)*NE
            isInCellDof = ~(isEdgeDof[:,0] | isEdgeDof[:,1] | isEdgeDof[:,2])
            idof = ldof - 3*p
            cell2dof[:, isInCellDof] = base + np.arange(NC*idof).reshape(NC, idof)

        return cell2dof

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.entity('cell')
        node = mesh.entity('node')

        if p == mesh.p:
            return node

        NN = mesh.number_of_nodes()
        GD = mesh.geo_dimension()
        gdof = self.number_of_global_dofs()
        ipoint = np.zeros((gdof, GD), dtype=np.float64)
        bcs = self.multiIndex/p 
        ipoint[cell2dof] = mesh.bc_to_point(bcs).swapaxes(0, 1)

        return ipoint

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh

        if p == mesh.p:
            return mesh.number_of_nodes()

        gdof = mesh.number_of_corner_nodes() # 注意这里只是单元角点的个数
        if p > 1:
            NE = self.mesh.number_of_edges()
            gdof += (p-1)*NE

        if p > 2:
            ldof = self.number_of_local_dofs()
            NC = self.mesh.number_of_cells()
            gdof += (ldof - 3*p)*NC
        return gdof

    def number_of_local_dofs(self, doftype='cell'):
        p = self.p
        if doftype in {'cell', 2}:
            return (p+1)*(p+2)//2 
        elif doftype in {'face', 'edge',  1}:
            return self.p + 1
        elif doftype in {'node', 0}:
            return 1
