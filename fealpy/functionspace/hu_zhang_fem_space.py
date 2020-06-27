import numpy as np
from .Function import Function
from .lagrange_fem_space import LagrangeFiniteElementSpace

class HuZhangFiniteElementSpace2d():
    """
    Hu-Zhang Mixed Finite Element Space.
    """
    def __init__(self, mesh, p):
        self.space = LagrangeFiniteElementSpace(mesh, p) # the scalar space
        self.mesh = mesh
        self.p = p
        self.dof = self.space.dof
        self.dim = self.space.dim
        self.cell2dof = self.init_cell_to_dof()

        t = mesh.edge_unit_tagent() 
        _, _, self.frame = np.linalg.svd(t[:, np.newaxis, :]) # get the axis frame on the edge by svd

        self.frame[:, 0, :] = t

    def __str__(self):
        return "2D Hu-Zhang mixed finite element space!"

    def number_of_global_dofs(self):
        """
        """
        p = self.p
        gdim = self.geo_dimension()
        tdim = self.tensor_dimension() 

        mesh = self.mesh

        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        gdof = tdim*NN

        if p > 1:
            edof = p - 1
            NE = mesh.number_of_edges()
            gdof += (tdim-1)*edof*NE # 边内部连续自由度的个数 
            E = mesh.number_of_edges_of_cells() # 单元边的个数
            gdof += NC*E*edof # 边内部不连续自由度的个数 

        if p > 2:
            cdof = (p+1)*(p+2)//2 - 3*p # 面内部自由度的个数
            gdof += tdim*cdof*NC

        return gdof 

    def number_of_local_dofs(self):
        tdim = self.tensor_dimension() 
        ldof = self.dof.number_of_local_dofs()
        return tdim*ldof

    def cell_to_dof(self):
        return self.cell2dof

    def init_cell_to_dof(self):
        """
        构建局部自由度到全局自由度的映射矩阵

        Returns
        -------
        cell2dof : ndarray with shape (NC, ldof*tdim)
            NC: 单元个数
            ldof: p 次标量空间局部自由度的个数
            tdim: 对称张量的维数
        """
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        gdim = self.geo_dimension()
        tdim = self.tensor_dimension() # 张量维数
        p = self.p
        dof = self.dof # 标量空间自由度管理对象 
       
        c2d = dof.cell2dof[..., np.newaxis]
        ldof = dof.number_of_local_dofs() # ldof : 标量空间单元上自由度个数
        cell2dof = np.zeros((NC, ldof, tdim), dtype=np.int) # 每个标量自由度变成 tdim 个自由度

        dofFlags = self.dof_flags_1() # 把不同类型的自由度区分开来
        idx, = np.nonzero(dofFlags[0]) # 局部顶点自由度的编号
        cell2dof[:, idx, :] = tdim*c2d[:, idx] + np.arange(tdim)

        base0 = 0
        base1 = 0
        idx, = np.nonzero(dofFlags[1]) # 边内部自由度的编号
        if len(idx) > 0:
            base0 += NN # 这是标量编号的新起点
            base1 += tdim*NN # 这是张量自由度编号的新起点
            #  0号局部自由度对应的是切向不连续的自由度, 留到后面重新编号
            cell2dof[:, idx, 1:] = base1 + (tdim-1)*(c2d[:, idx] - base0) + np.arange(tdim - 1)

        idx, = np.nonzero(dofFlags[2])
        if len(idx) > 0:
            edof = p - 1
            base0 += edof*NE
            base1 += (tdim-1)*edof*NE
            cell2dof[:, idx, :] = base1 + tdim*(c2d[:, idx] - base0) + np.arange(tdim)

        cdof = (p+1)*(p+2)//2 - 3*p # 边内部自由度

        idx, = np.nonzero(dofFlags[1])
        if len(idx) > 0:
            base1 += tdim*cdof*NC 
            cell2dof[:, idx, 0] = base1 + np.arange(NC*len(idx)).reshape(NC, len(idx)) 

        self.cell2dof = cell2dof.reshape(NC, -1)

    def dof_flags(self):
        """ 对标量空间中的自由度进行分类, 分为边内部自由度, 面内部自由度(如果是三维空间的话)及其它自由度 

        Returns
        -------

        isOtherDof : ndarray, (ldof,)
            除了边内部和面内部自由度的其它自由度
        isEdgeDof : ndarray, (ldof, 3) 
            每个边内部的自由度
        -------

        """
        dim = self.geo_dimension()
        dof = self.dof 
        
        isPointDof = dof.is_on_node_local_dof()
        isEdgeDof = dof.is_on_edge_local_dof()
        isEdgeDof[isPointDof] = False
        
        isEdgeDof0 = np.sum(isEdgeDof, axis=-1) > 0 # 
        isOtherDof = (~isEdgeDof0) # 除了边内部自由度之外的其它自由度
                                   # dim = 2: 包括点和面内部自由度
        return isOtherDof, isEdgeDof

    def dof_flags_1(self):
        """ 
        对标量空间中的自由度进行分类, 分为:
            点上的自由由度
            边内部的自由度
            面内部的自由度

        Returns
        -------

        """
        gdim = self.geo_dimension() # the geometry space dimension
        dof = self.dof 
        isPointDof = dof.is_on_node_local_dof()
        isEdgeDof = dof.is_on_edge_local_dof()
        isEdgeDof[isPointDof] = False
        isEdgeDof0 = np.sum(isEdgeDof, axis=-1) > 0
        return isPointDof, isEdgeDof0, ~(isPointDof | isEdgeDof0)

    def geo_dimension(self):
        return self.dim

    def tensor_dimension(self):
        dim = self.dim
        return dim*(dim - 1)//2 + dim

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def basis(self, bc, cellidx=None):
        """
        Parameters
        ----------
        bc : ndarray with shape (NQ, dim+1)
            bc[i, :] is i-th quad point
        cellidx : ndarray
            有时我我们只需要计算部分单元上的基函数
        Returns
        -------
        phi : ndarray with shape (NQ, NC, ldof*tdim, 3 or 6)
            NQ: 积分点个数
            NC: 单元个数
            ldof: 标量空间的单元自由度个数
            tdim: 对称张量的维数
        """
        mesh = self.mesh

        gdim = self.geo_dimension() 
        tdim = self.tensor_dimension()

        if cellidx is None:
            NC = mesh.number_of_cells()
            cell2edge = mesh.ds.cell_to_edge()
        else:
            NC = len(cellidx)
            cell2edge = mesh.ds.cell_to_edge()[cellidx]

        phi0 = self.space.basis(bc) # the shape of phi0 is (NQ, ldof)
        shape = list(phi0.shape)
        shape.insert(-1, NC)
        shape += [tdim, tdim]
        # The shape of `phi` is (NQ, NC, ldof, tdim, tdim), where
        #   NQ : the number of quadrature points 
        #   NC : the number of cells
        #   ldof : the number of dofs in each cell
        #   tdim : the dimension of symmetric tensor matrix
        phi = np.zeros(shape, dtype=np.float) 

        dofFlag = self.dof_flags()
        # the dof on the vertex and the interior of the cell
        isOtherDof = dofFlag[0]
        idx, = np.nonzero(isOtherDof)
        if len(idx) > 0:
            phi[..., idx[..., np.newaxis], range(tdim), range(tdim)] = phi0[..., np.newaxis, idx, np.newaxis]
  
        isEdgeDof = dofFlag[1]
        for i, isDof in enumerate(isEdgeDof.T):
            phi[..., isDof, :, :] = np.einsum('...j, imn->...ijmn', phi0[..., isDof], self.TE[cell2edge[:, i]]) 

        # The shape of `phi` should be (NQ, NC, ldof*tdim, tdim)?
        shape = phi.shape[:-3] + (-1, tdim)
        return phi.reshape(shape)
