import numpy as np
from .function import Function

from fealpy.functionspace import LagrangeFiniteElementSpace
from ..quadrature import  IntervalQuadrature

class HuZhangFiniteElementSpace():
    """
    Hu-Zhang Mixed Finite Element Space.
    """
    def __init__(self, mesh, p):
        self.space = LagrangeFiniteElementSpace(mesh, p) # the scalar space
        self.mesh = mesh
        self.p = p
        self.dof = self.space.dof
        self.dim = self.space.GD
        self.init_orth_matrices()
        self.init_cell_to_dof()

    def init_orth_matrices(self):
        """
        Initialize the othogonal symetric matrix basis.
        """
        mesh = self.mesh
        gdim = self.geo_dimension()

        NE = mesh.number_of_edges()
        if gdim == 2:
            idx = np.array([(0, 0), (1, 1), (0, 1)])
            self.TE = np.zeros((NE, 3, 3), dtype=np.float)
            self.T = np.array([[(1, 0), (0, 0)], [(0, 0), (0, 1)], [(0, 1), (1, 0)]])
        elif gdim == 3:
            idx = np.array([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)])
            self.TE = np.zeros((NE, 6, 6), dtype=np.float)
            self.T = np.array([
                [(1, 0, 0), (0, 0, 0), (0, 0, 0)], 
                [(0, 0, 0), (0, 1, 0), (0, 0, 0)],
                [(0, 0, 0), (0, 0, 0), (0, 0, 1)],
                [(0, 0, 0), (0, 0, 1), (0, 1, 0)],
                [(0, 0, 1), (0, 0, 0), (1, 0, 0)],
                [(0, 1, 0), (1, 0, 0), (0, 0, 0)]])

        t = mesh.edge_unit_tagent() 
        _, _, frame = np.linalg.svd(t[:, np.newaxis, :]) # get the axis frame on the edge by svd
        frame[:, 0, :] = t
        for i, (j, k) in enumerate(idx):
            self.TE[:, i] = (frame[:, j, idx[:, 0]]*frame[:, k, idx[:, 1]] + frame[:, j, idx[:, 1]]*frame[:, k, idx[:, 0]])/2
        self.TE[:, gdim:] *=np.sqrt(2) 

        if gdim == 3:
            NF = mesh.number_of_faces()
            self.TF = np.zeros((NF, 6, 6), dtype=np.float)
            n = mesh.face_unit_normal()
            _, _, frame = np.linalg.svd(n[:, np.newaxis, :]) # get the axis frame on the face by svd
            frame[:, 0, :] = n 
            for i, (j, k) in enumerate(idx):
                self.TF[:, i] = (frame[:, j, idx[:, 0]]*frame[:, k, idx[:, 1]] + frame[:, j, idx[:, 1]]*frame[:, k, idx[:, 0]])/2

            self.TF[:, gdim:] *= np.sqrt(2)

    def __str__(self):
        return "Hu-Zhang mixed finite element space!"

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
            fdof = (p+1)*(p+2)//2 - 3*p # 面内部自由度的个数
            if gdim == 2:
                gdof += tdim*fdof*NC
            elif gdim == 3:
                NF = mesh.number_of_faces()
                gdof += 3*fdof*NF # 面内部连续自由度的个数
                F = mesh.number_of_faces_of_cells() # 每个单元面的个数
                gdof += 3*F*fdof*NC # 面内部不连续自由度的个数

        if (p > 3) and (gdim == 3):
            ldof = self.dof.number_of_local_dofs()
            V = mesh.number_of_nodes_of_cells() # 单元顶点的个数
            cdof = ldof - E*edof - F*fdof - V 
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
            if gdim == 2:
                cell2dof[:, idx, :] = base1 + tdim*(c2d[:, idx] - base0) + np.arange(tdim)
            elif gdim == 3:
                # 1, 2, 3 号局部自由度对应切向不连续的张量自由度, 留到后面重新编号
                # TODO: check it is right
                cell2dof[:, idx.reshape(-1, 1), np.array([0, 4, 5])]= base1 + (tdim - 3)*(c2d[:, idx] - base0) + np.arange(tdim - 3)

        fdof = (p+1)*(p+2)//2 - 3*p # 边内部自由度
        if gdim == 3:
            idx, = np.nonzero(dofFlags[3])
            if len(idx) > 0:
                NF = mesh.number_of_faces()
                base0 += fdof*NF 
                base1 += (tdim - 3)*fdof*NF
                cell2dof[:, idx, :] = base1 + tdim*(c2d[:, idx] - base0) + np.arange(tdim)
            cdof = ldof - 4*fdof - 6*edof - 4 # 单元内部自由度
        else:
            cdof = fdof

        idx, = np.nonzero(dofFlags[1])
        if len(idx) > 0:
            base1 += tdim*cdof*NC 
            cell2dof[:, idx, 0] = base1 + np.arange(NC*len(idx)).reshape(NC, len(idx)) 

        if gdim == 3:
            base1 += NC*len(idx)
            idx, = np.nonzero(dofFlags[2])
            print(idx)
            if len(idx) > 0:
                cell2dof[:, idx.reshape(-1, 1), np.array([1, 2, 3])] = base1 + np.arange(NC*len(idx)*3).reshape(NC, len(idx), 3)

        self.cell2dof = cell2dof.reshape(NC, -1)

    def geo_dimension(self):
        return self.dim

    def tensor_dimension(self):
        dim = self.dim
        return dim*(dim - 1)//2 + dim

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def dof_flags(self):
        """ 对标量空间中的自由度进行分类, 分为边内部自由度, 面内部自由度(如果是三维空间的话)及其它自由度 

        Returns
        -------

        isOtherDof : ndarray, (ldof,)
            除了边内部和面内部自由度的其它自由度
        isEdgeDof : ndarray, (ldof, 3) or (ldof, 6) 
            每个边内部的自由度
        isFaceDof : ndarray, (ldof, 4)
            每个面内部的自由度
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
                                   # dim = 3: 包括点, 面内部和体内部自由度
        if dim == 2:
            return isOtherDof, isEdgeDof
        elif dim == 3:
            isFaceDof = dof.is_on_face_local_dof()
            isFaceDof[isPointDof, :] = False
            isFaceDof[isEdgeDof0, :] = False

            isFaceDof0 = np.sum(isFaceDof, axis=-1) > 0
            isOtherDof = isOtherDof & (~isFaceDof0) # 三维情形下, 从其它自由度中除去面内部自由度

            return isOtherDof, isEdgeDof, isFaceDof
        else:
            raise ValueError('`dim` should be 2 or 3!')

    def dof_flags_1(self):
        """ 
        对标量空间中的自由度进行分类, 分为:
            点上的自由由度
            边内部的自由度
            面内部的自由度
            体内部的自由度

        Returns
        -------

        """
        gdim = self.geo_dimension() # the geometry space dimension
        dof = self.dof 
        isPointDof = dof.is_on_node_local_dof()
        isEdgeDof = dof.is_on_edge_local_dof()
        isEdgeDof[isPointDof] = False
        isEdgeDof0 = np.sum(isEdgeDof, axis=-1) > 0
        if gdim == 2:
            return isPointDof, isEdgeDof0, ~(isPointDof | isEdgeDof0)
        elif gdim == 3:
            isFaceDof = dof.is_on_face_local_dof()
            isFaceDof[isPointDof, :] = False
            isFaceDof[isEdgeDof0, :] = False

            isFaceDof0 = np.sum(isFaceDof, axis=-1) > 0
            return isPointDof, isEdgeDof0, isFaceDof0, ~(isPointDof | isEdgeDof0 | isFaceDof0)
        else:
            raise ValueError('`dim` should be 2 or 3!')

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

        if gdim == 3:
            if cellidx is None:
                cell2face = mesh.ds.cell_to_face()
            else:
                cell2face = mesh.ds.cell_to_face()[cellidx]
            isFaceDof = dofFlag[2]
            for i, isDof in enumerate(isFaceDof.T):
                phi[..., isDof, :, :] = np.einsum('...j, imn->...ijmn', phi0[..., isDof], self.TF[cell2face[:, i]])
        # The shape of `phi` should be (NQ, NC, ldof*tdim, tdim)?
        shape = phi.shape[:-3] + (-1, tdim)
        return phi.reshape(shape)

    def div_basis(self, bc, cellidx=None):
        mesh = self.mesh

        gdim = self.geo_dimension()
        tdim = self.tensor_dimension() 

        # the shape of `gphi` is (NQ, NC, ldof, gdim)
        gphi = self.space.grad_basis(bc, cellidx=cellidx) 
        shape = list(gphi.shape)
        shape.insert(-1, tdim)
        # the shape of `dphi` is (NQ, NC, ldof, tdim, gdim)
        dphi = np.zeros(shape, dtype=np.float)

        dofFlag = self.dof_flags()
        # the dof on the vertex and the interior of the cell
        isOtherDof = dofFlag[0]
        dphi[..., isOtherDof, :, :] = np.einsum('...ijm, kmn->...ijkn', gphi[..., isOtherDof, :], self.T)

        if cellidx is None:
            cell2edge = mesh.ds.cell_to_edge()
        else:
            cell2edge = mesh.ds.cell_to_edge()[cellidx]
        isEdgeDof = dofFlag[1]
        for i, isDof in enumerate(isEdgeDof.T):
            VAL = np.einsum('ijk, kmn->ijmn', self.TE[cell2edge[:, i]], self.T)
            dphi[..., isDof, :, :] = np.einsum('...ikm, ijmn->...ikjn', gphi[..., isDof, :], VAL) 

        if gdim == 3:
            if cellidx is None:
                cell2face = mesh.ds.cell_to_face()
            else:
                cell2face = mesh.ds.cell_to_face()[cellidx]
            isFaceDof = dofFlag[2]
            for i, isDof in enumerate(isFaceDof.T):
                VAL = np.einsum('ijk, kmn->ijmn', self.TF[cell2face[:, i]], self.T)
                dphi[..., isDof, :, :] = np.einsum('...ikm, ijmn->...ikjn', gphi[..., isDof, :], VAL) 

        # The new shape of `dphi` is `(NQ, NC, ldof*tdim, gdim)`, where
        shape = dphi.shape[:-3] + (-1, gdim)
        return dphi.reshape(shape)

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc, cellidx=cellidx)
        cell2dof = self.cell_to_dof()
        tdim = self.tensor_dimension()
        if cellidx is None:
            uh = uh[cell2dof]
        else:
            uh = uh[cell2dof[cellidx]]
        phi = np.einsum('...jk, kmn->...jmn', phi, self.T)
        val = np.einsum('...ijmn, ij->...imn', phi, uh) 
        return val 

    def div_value(self, uh, bc, cellidx=None):
        dphi = self.div_basis(bc, cellidx=cellidx)
        cell2dof = self.cell_to_dof()
        tdim = self.tensor_dimension()
        if cellidx is None:
            uh = uh[cell2dof]
        else:
            uh = uh[cell2dof[cellidx]]
        val = np.einsum('...ijm, ij->...im', dphi, uh)
        return val

    def interpolation(self, u):

        mesh = self.mesh;
        gdim = self.geo_dimension()
        tdim = self.tensor_dimension()

        if gdim == 2:
            idx = np.array([(0, 0), (1, 1), (0, 1)])
        elif gdim == 3:
            idx = np.array([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)])

        ipoint = self.dof.interpolation_points()
        c2d = self.dof.cell2dof
        val = u(ipoint)[c2d]

        ldof = self.dof.number_of_local_dofs()
        cell2dof = self.cell2dof.reshape(-1, ldof, tdim)

        uI = Function(self)
        dofFlag = self.dof_flags()
        isOtherDof = dofFlag[0]
        idx0, = np.nonzero(isOtherDof)
        uI[cell2dof[:, idx0, :]] = val[:, idx0][..., idx[:, 0], idx[:, 1]]

        isEdgeDof = dofFlag[1]
        cell2edge = self.mesh.ds.cell_to_edge()
        for i, isDof in enumerate(isEdgeDof.T):
            TE = np.einsum('ijk, kmn->ijmn', self.TE[cell2edge[:, i]], self.T)
            uI[cell2dof[:, isDof, :]] = np.einsum('ikmn, ijmn->ikj', val[:, isDof, :, :], TE)

        if gdim == 3:
            cell2face = mesh.ds.cell_to_face()
            isFaceDof = dofFlag[2]
            for i, isDof in enumerate(isFaceDof.T):
                TF = np.einsum('ijk, kmn->ijmn', self.TF[cell2face[:, i]], self.T)
                uI[cell2dof[:, isDof, :]] = np.einsum('ikmn, ijmn->ikj', val[..., isDof, :, :], TF) 
        return uI
 
        def function(self, dim=None):
            f = Function(self)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        return np.zeros(gdof, dtype=np.float)


class RTFiniteElementSpace2d:
    def __init__(self, mesh, p=0):
        self.mesh = mesh
        self.p = p

    def cell_to_edge_sign(self):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        edge2cell = mesh.ds.edge2cell
        cell2edgeSign = -np.ones((NC, 3), dtype=np.int)
        cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = 1
        return cell2edgeSign

    def basis(self, bc):
        mesh = self.mesh
        p = self.p
        ldof = self.number_of_local_dofs()
        NC = mesh.number_of_cells()
        Rlambda = mesh.rot_lambda()
        cell2edgeSign = self.cell_to_edge_sign()
        shape = bc.shape[:-1] + (NC, ldof, 2)
        phi = np.zeros(shape, dtype=np.float)
        if p == 0:
            phi[..., 0, :] = bc[..., 1, np.newaxis, np.newaxis]*Rlambda[:, 2, :] - bc[..., 2, np.newaxis, np.newaxis]*Rlambda[:, 1, :]
            phi[..., 1, :] = bc[..., 2, np.newaxis, np.newaxis]*Rlambda[:, 0, :] - bc[..., 0, np.newaxis, np.newaxis]*Rlambda[:, 2, :]
            phi[..., 2, :] = bc[..., 0, np.newaxis, np.newaxis]*Rlambda[:, 1, :] - bc[..., 1, np.newaxis, np.newaxis]*Rlambda[:, 0, :]
            phi *= cell2edgeSign.reshape(-1, 3, 1)
        elif p == 1:
            pass
        else:
            raise ValueError('p')

        return phi

    def grad_basis(self, bc):
        mesh = self.mesh
        p = self.p

        ldof = self.number_of_local_dofs()
        NC = mesh.number_of_cells()
        shape = (NC, ldof, 2, 2)
        gradPhi = np.zeros(shape, dtype=np.float)

        cell2edgeSign = self.cell_to_edge_sign()
        W = np.array([[0, 1], [-1, 0]], dtype=np.float)
        Rlambda= mesh.rot_lambda()
        Dlambda = mesh.grad_lambda()
        if p == 0:
            A = np.einsum('...i, ...j->...ij', Rlambda[:, 2, :], Dlambda[:, 1, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 1, :], Dlambda[:, 2, :]) 
            gradPhi[:, 0, :, :] = A - B

            A = np.einsum('...i, ...j->...ij', Rlambda[:, 0, :], Dlambda[:, 2, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 2, :], Dlambda[:, 0, :])
            gradPhi[:, 1, :, :] = A - B

            A = np.einsum('...i, ...j->...ij', Rlambda[:, 1, :], Dlambda[:, 0, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 0, :], Dlambda[:, 1, :])
            gradPhi[:, 2, :, :] = A - B

            gradPhi *= cell2edgeSign.reshape(-1, 3, 1, 1)
        else:
            #TODO:raise a error
            print("error")

        return gradPhi 

    def div_basis(self, bc):
        mesh = self.mesh
        p = self.p

        ldof = self.number_of_local_dofs()
        NC = mesh.number_of_cells()
        divPhi = np.zeros((NC, ldof), dtype=np.float)
        cell2edgeSign = self.cell_to_edge_sign()
        W = np.array([[0, 1], [-1, 0]], dtype=np.float)

        Rlambda = mesh.rot_lambda()
        Dlambda = mesh.grad_lambda()
        if p == 0:
            divPhi[:, 0] = np.sum(Dlambda[:, 1, :]*Rlambda[:, 2, :], axis=1) - np.sum(Dlambda[:, 2, :]*Rlambda[:, 1, :], axis=1)
            divPhi[:, 1] = np.sum(Dlambda[:, 2, :]*Rlambda[:, 0, :], axis=1) - np.sum(Dlambda[:, 0, :]*Rlambda[:, 2, :], axis=1)
            divPhi[:, 2] = np.sum(Dlambda[:, 0, :]*Rlambda[:, 1, :], axis=1) - np.sum(Dlambda[:, 1, :]*Rlambda[:, 0, :], axis=1)
            divPhi *= cell2edgeSign
        else:
            #TODO:raise a error
            print("error")

        return divPhi

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        if p == 0:
            cell2dof = mesh.ds.cell_to_edge()
        else:
            #TODO: raise a error 
            print('error!')

        return cell2dof

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        if p == 0:
            return NE
        else:
            #TODO: raise a error
            print("error!")

    def number_of_local_dofs(self):
        p = self.p
        if p==0:
            return 3
        else:
            print("error!")

    def value(self, uh, bc, cellidx=None):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, phi, uh[cell2dof])
        else:
            val = np.einsum(s1, phi, uh[cell2dof[cellidx]])
        return val

    def grad_value(self, uh, bc, cellidx=None):
        gphi = self.grad_basis(bc, cellidx=cellidx)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijmn, ij{}->...i{}mn'.format(s0[:dim], s0[:dim])
        if cellidx is None:
            val = np.einsum(s1, gphi, uh[cell2dof])
        else:
            val = np.einsum(s1, gphi, uh[cell2dof[cellidx]])
        return val

    def div_value(self, uh, bc, cellidx=None):
        val = self.grad_value(uh, bc, cellidx=None)
        return val.trace(axis1=-2, axis2=-1)

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def interpolation(self, u, returnfun=False):
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        NE = mesh.number_of_edges()
        n = mesh.edge_unit_normal()
        l = mesh.entity_measure('edge')

        qf = IntervalQuadrature(3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        points = np.einsum('kj, ijm->kim', bcs, node[edge])
        val = u(points)
        uh = np.einsum('k, kim, im, i->i', ws, val, n, l)

        if returnfun is True:
            return Function(self, array=uh)
        else:
            return uh

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

class BDMFiniteElementSpace2d:
    def __init__(self, mesh, p=1, dtype=np.float):
        self.mesh = mesh
        self.p = p
        self.dtype= dtype

    def cell_to_edge_sign(self):
        mesh = self.mesh
        edge2cell = mesh.ds.edge2cell
        NC = mesh.number_of_cells()
        cell2edgeSign = -np.ones((NC, 3), dtype=np.int)
        cell2edgeSign[edge2cell[:, 0], edge2cell[:, 2]] = 1
        return cell2edgeSign

    def basis(self, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()

        ldof = self.number_of_local_dofs()

        NC = mesh.number_of_cells()
        p = self.p
        phi = np.zeros((NC, ldof, dim), dtype=self.dtype)

        cell2edgeSign = self.cell_to_edge_sign()
        Rlambda, _ = mesh.rot_lambda()
        if p == 1:
            phi[:, 0, :] = bc[1]*Rlambda[:, 2, :] - bc[2]*Rlambda[:, 1, :]
            phi[:, 1, :] = bc[1]*Rlambda[:, 2, :] + bc[2]*Rlambda[:, 1, :]

            phi[:, 2, :] = bc[2]*Rlambda[:, 0, :] - bc[0]*Rlambda[:, 2, :]
            phi[:, 3, :] = bc[2]*Rlambda[:, 0, :] + bc[0]*Rlambda[:, 2, :]

            phi[:, 4, :] = bc[0]*Rlambda[:, 1, :] - bc[1]*Rlambda[:, 0, :]
            phi[:, 5, :] = bc[0]*Rlambda[:, 1, :] + bc[1]*Rlambda[:, 0, :]

            phi[:, 0:6:2, :] *=cell2edgeSign.reshape(-1, 3, 1)
        else:
            #TODO:raise a error
            print("error")

        return phi

    def grad_basis(self, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()
        p = self.p

        gradPhi = np.zeros((NC, ldof, dim, dim), dtype=self.dtype)

        cell2edgeSign = self.cell_to_edge_sign()
        W = np.array([[0, 1], [-1, 0]], dtype=self.dtype)
        Rlambda, _ = mesh.rot_lambda()
        Dlambda = Rlambda@W
        if p == 1:
            A = np.einsum('...i, ...j->...ij', Rlambda[:, 2, :], Dlambda[:, 1, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 1, :], Dlambda[:, 2, :]) 
            gradPhi[:, 0, :, :] = A - B 
            gradPhi[:, 1, :, :] = A + B

            A = np.einsum('...i, ...j->...ij', Rlambda[:, 0, :], Dlambda[:, 2, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 2, :], Dlambda[:, 0, :])
            gradPhi[:, 2, :, :] = A - B
            gradPhi[:, 3, :, :] = A + B

            A = np.einsum('...i, ...j->...ij', Rlambda[:, 1, :], Dlambda[:, 0, :])
            B = np.einsum('...i, ...j->...ij', Rlambda[:, 0, :], Dlambda[:, 1, :])
            gradPhi[:, 4, :, :] = A - B
            gradPhi[:, 5, :, :] = A + B

            gradPhi[:, 0:6:2, :, :] *= cell2edgeSign.reshape(-1, 3, 1, 1) 
            gradPhi[:, 1:6:2, :, :] *= cell2edgeSign.reshape(-1, 3, 1, 1) 
        else:
            #TODO:raise a error
            print("error")

        return gradPhi 

    def div_basis(self, bc):
        mesh = self.mesh
        p = self.p

        divPhi = np.zeors((NC, ldof), dtype=self.dtype)

        Dlambda, _ = mesh.grad_lambda()
        Rlambda, _ = mesh.rot_lambda()
        if p == 1:
            divPhi[:, 0] = np.sum(Dlambda[:, 1, :]*Rlambda[:, 2, :] - Dlambda[:, 2, :]*Rlambda[:, 1, :], axia=1)
            divPhi[:, 1] = np.sum(Dlambda[:, 1, :]*Rlambda[:, 2, :] + Dlambda[:, 2, :]*Rlambda[:, 1, :], axia=1)
            divPhi[:, 2] = np.sum(Dlambda[:, 2, :]*Rlambda[:, 0, :] - Dlambda[:, 0, :]*Rlambda[:, 2, :], axis=1)
            divPhi[:, 3] = np.sum(Dlambda[:, 2, :]*Rlambda[:, 0, :] + Dlambda[:, 0, :]*Rlambda[:, 2, :], axis=1)
            divPhi[:, 4] = np.sum(Dlambda[:, 0, :]*Rlambda[:, 1, :] - Dlambda[:, 1, :]*Rlambda[:, 0, :], axis=1)
            divPhi[:, 5] = np.sum(Dlambda[:, 0, :]*Rlambda[:, 1, :] + Dlambda[:, 1, :]*Rlambda[:, 0, :], axis=1)
            divPhi[:, 0:6:2] *= cell2edgeSign
            divPhi[:, 1:6:2] *= cell2edgeSign
        else:
            #TODO:raise a error
            print("error")

        return divPhi 

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NE = mesh.number_of_edges()

        if p == 1:
            edge2dof = np.arange(2*NE).reshape(NE, 2)
        else:
            #TODO: raise error
            print('error!')

        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        edge2dof = self.edge_to_dof()

        cell2edgeSign = mesh.ds.cell_to_edge_sign()
        cell2edge = mesh.ds.cell_to_edge()

        if p == 1:
            cell2dof = np.zeros((NC, ldof), dtype=np.int)
            cell2dof[cell2edgeSign[:, 0], 0:2]= edge2dof[cell2edge[cell2edgeSign[:, 0], 0], :]  
            cell2dof[~cell2edgeSign[:, 0], 0:2]= edge2dof[cell2edge[~cell2edgeSign[:, 0], 0], -1::-1]  

            cell2dof[cell2edgeSign[:, 1], 2:4]= edge2dof[cell2edge[cell2edgeSign[:, 1], 1], :]  
            cell2dof[~cell2edgeSign[:, 1], 2:4]= edge2dof[cell2edge[~cell2edgeSign[:, 1], 1], -1::-1]  

            cell2dof[cell2edgeSign[:, 2], 4:6]= edge2dof[cell2edge[cell2edgeSign[:, 2], 2], :]  
            cell2dof[~cell2edgeSign[:, 2], 4:6]= edge2dof[cell2edge[~cell2edgeSign[:, 2], 2], -1::-1]  

        return cell2dof

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        if p == 1:
            return 2*NE
        else:
            #TODO: raise a error
            print("error!")

    def number_of_local_dofs(self):
        p = self.p
        if p == 1:
            return 6
        else:
            #TODO: raise a error
            print("error!")


class RaviartThomasFiniteElementSpace3d:
    def __init__(self, mesh, p=0, dtype=np.float):
        self.mesh = mesh
        self.p = p
        self.dtype= dtype

    def basis(self, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()

        ldof = self.number_of_local_dofs()

        p = self.p
        phi = np.zeors((NC, ldof, dim), dtype=self.dtype)


        return phi

    def grad_basis(self, bc):
        mesh = self.mesh
        dim = mesh.geom_dimension()
        p = self.p

        gradPhi = np.zeros((NC, ldof, dim, dim), dtype=self.dtype)

        return gradPhi 

    def div_basis(self, bc):
        mesh = self.mesh
        p = self.p

        divPhi = np.zeors((NC, ldof), dtype=self.dtype)

        return divPhi 

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell

        N = mesh.number_of_nodes()
        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()

        return cell2dof

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        if p == 0:
            return NE
        elif p==1:
            return 2*NE
        else:
            #TODO: raise a error
            print("error!")


    def number_of_local_dofs(self):
        p = self.p
        if p==0:
            return 3
        elif p==1:
            return 6
        else:
            #TODO: raise a error
            print("error!")


