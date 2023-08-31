import numpy as np
from numpy.linalg import inv

from scipy.sparse import csr_matrix, coo_matrix
from ..decorator import barycentric
from .Function import Function
from ..quadrature import FEMeshIntegralAlg
from ..decorator import timer
from scipy.sparse.linalg import spsolve, cg

class FNDof3d:
    def __init__(self, mesh):
        """
        Parameters
        ----------
        mesh : TetrahedronMesh object

        Notes
        -----

        Reference
        ---------
        """
        self.mesh = mesh
        self.cell2dof = self.cell_to_dof() # 默认的自由度数组

    def boundary_dof(self, threshold=None):
        return self.mesh.ds.boundary_edge_flag()

    def is_boundary_dof(self, threshold=None):
        return self.mesh.ds.boundary_edge_flag()

    def edge_to_dof(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        return np.arange(NE) 

    def face_to_dof(self, index=np.s_[:]):
        mesh = self.mesh
        return mesh.ds.face_to_edge()[index]

    def cell_to_dof(self):
        mesh = self.mesh
        return mesh.ds.cell_to_edge()

    def number_of_local_dofs(self, doftype='all'):
        stype = self.spacetype
        if doftype == 'all': # number of all dofs on a cell 
            return 6
        elif doftype in {'cell', 3}: # number of dofs inside the cell 
            return 0 
        elif doftype in {'face', 2}: # number of dofs on a face 
            return 1
        elif doftype in {'edge', 1}: # number of dofs on a edge
            return 1
        elif doftype in {'node', 0}: # number of dofs on a node
            return 0

    def number_of_global_dofs(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        return NE

class FirstNedelecFiniteElementSpace3d:
    def __init__(self, mesh, p = 1, q=None, dof=None):
        """
        """
        self.p = p
        self.mesh = mesh

        if dof is None:
            self.dof = FNDof3d(mesh)
        else:
            self.dof = dof

        self.integralalg = FEMeshIntegralAlg(self.mesh, 2)
        self.integrator = self.integralalg.integrator

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype

    def boundary_dof(self):
        return self.dof.boundary_dof()

    @barycentric
    def face_basis(self, bc, index=np.s_[:], barycenter=True):
        """!
        @brief 因为基函数在面上法向不连续，切向连续，所以认为面上的基函数只有切向分量。
        """
        p = self.p
        mesh = self.mesh
        localEdge = np.array([[1, 2], [2, 0], [0, 1]], dtype=np.int_)

        n = mesh.face_unit_normal()[index] #(NF, 3)
        fm = mesh.entity_measure("face")[index]
        face = mesh.entity("face")[index]
        node = mesh.entity("node")
        e = node[face[:, localEdge[:, 1]]] - node[face[:, localEdge[:, 0]]] #(NF, 3, 3)
        glambda = np.cross(n[:, None], e)/(2*fm[:, None, None]) #(NF, 3, 3)
        if p==1:
            fphi = bc[..., None, localEdge[:, 0], None]*glambda[:, 
                    localEdge[:, 1]] - bc[..., None, localEdge[:, 1], None]*glambda[:,
                    localEdge[:, 0]]
            f2es = mesh.ds.face_to_edge_sign().astype(np.int_)[index]
            f2es[f2es==0] = -1
            return fphi*f2es[..., None]

    @barycentric
    def edge_basis(self, bc, index=None, barycenter=True, left=True):
        pass

    @barycentric
    def basis(self, bc, index=np.s_[:]):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(4,)` or `(NQ, 4)`
        Returns
        -------
        phi : numpy.ndarray
            the shape of 'phi' can be `(NC, ldof, 3)` or `(NQ, NC, ldof, 3)`

        See Also
        --------

        Notes
        -----
        (NC, NQ, ldof, 3)
        (NC, ldof, NQ, 3)
        """

        p = self.p
        mesh = self.mesh
        if p == 1:
            localEdge = mesh.ds.localEdge
            glambda = mesh.grad_lambda()[index] # (NC, 4, 3)
            phi = bc[..., None, localEdge[:, 0], None]*glambda[:, 
                    localEdge[:, 1]] - bc[..., None, localEdge[:, 1], None]*glambda[:,
                    localEdge[:, 0]]
            c2es = mesh.ds.cell_to_edge_sign().astype(np.float_)[index]
            c2es[c2es==0] = -1
            return phi*c2es[..., None]
        else:
            pass

    @barycentric
    def curl_basis(self, bc, index=np.s_[:]):
        """

        Parameters
        ----------

        Notes
        -----

        """
        p = self.p
        mesh = self.mesh
        NC = mesh.number_of_cells()

        if p == 1:
            localEdge = mesh.ds.localEdge
            glambda = mesh.grad_lambda() # (NC, 4, 3)
            cphi = 2*np.cross(glambda[:, localEdge[:, 0]], glambda[:, localEdge[:, 1]])

            c2es = mesh.ds.cell_to_edge_sign().astype(np.float_)
            c2es[c2es==0] = -1
            cphi = cphi*c2es[..., None]

            shape = bc.shape[:-1] + cphi.shape
            cphi = np.broadcast_to(cphi, shape)
            return cphi
        else:
            pass

    @barycentric
    def grad_basis(self, bc, index=np.s_[:]):
        """

        Parameters
        ----------

        Notes
        -----

        """
        pass

    def cell_to_dof(self):
        return self.dof.cell2dof

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype)

    @barycentric
    def face_value(self, uh, bc, index=np.s_[:]):
        phi = self.face_basis(bc, index=index) #(..., NF, ldof, 3)
        face2dof = self.dof.face_to_dof() #(NF, ldof)
        s1 = '...ijm, ij->...im'
        val = np.einsum(s1, phi, uh[face2dof[index]])
        return val

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        phi = self.basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[cell2dof[index]])
        return val

    @barycentric
    def curl_value(self, uh, bc, index=np.s_[:]):
        cphi = self.curl_basis(bc, index=index)
        cell2dof = self.cell_to_dof()
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, cphi, uh[cell2dof[index]])
        return val

    @barycentric
    def edge_value(self, uh, bc, index=np.s_[:], left=True):
        phi = self.edge_basis(bc, index=index, left=left)
        edge2dof = self.dof.edge_to_dof()[index] 
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[edge2dof])
        return val

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        pass

    def function(self, dim=None, array=None, dtype=np.float64):
        return Function(self, dim=dim, array=array, coordtype='barycentric',
                dtype=dtype)

    def project(self, u):
        A = self.mass_matrix()
        b = self.source_vector(u)
        up = self.function()
        up[:] = spsolve(A, b)
        return up

    def interpolation(self, u):
        mesh = self.mesh
        node = mesh.entity("node")
        edge = mesh.entity("edge")

        et = mesh.edge_tangent()
        point = 0.5*node[edge[:, 0]] + 0.5*node[edge[:, 1]]

        uI = self.function()
        uI[:] = np.sum(u(point)*et, axis=-1)
        return uI

    def mass_matrix(self, c=None, q=None, dtype=np.float_):
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        phi = self.basis(bcs) #(NQ, NC, 6, 3)
        cellmeasure = self.mesh.cell_volume()
        if c is None:
            val = np.einsum('q, qckd, qcmd, c->ckm', ws, phi, phi, cellmeasure, optimize=True)
        else:
            if callable(c):
                if c.coordtype == 'barycentric':
                    c = c(bcs)
                elif c.coordtype == 'cartesian':
                    ps = self.mesh.bc_to_point(bcs)
                    c = c(ps)

            if isinstance(c, (int, float)):
                M = np.einsum('q, qcld, qcmd, c->clm', c*ws, phi, phi, cellmeasure, optimize=True)
            elif(len(c.shape)==4):
                M = np.einsum('q, qcnd, qckn, qcmd, c->ckm', ws, c, phi, phi, cellmeasure, optimize=True)
            else:
                M = np.einsum('q, qc, qckn, qcmd, c->ckm', ws, c, phi, phi, cellmeasure, optimize=True)

        cell2dof = self.cell_to_dof()
        I = np.broadcast_to(cell2dof[..., None], M.shape)
        J = np.broadcast_to(cell2dof[:, None, :], M.shape)
        gdof = self.dof.number_of_global_dofs()
        return csr_matrix((M.flat, (I.flat, J.flat)), shape = (gdof, gdof),
                dtype=dtype)

    def curl_matrix(self, c=None, q=None, dtype=np.float_):
        """

        Notes:

        组装 (c*\\nabla \\times u_h, \\nabla \\times u_h) 矩阵 
        """
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        phi = self.curl_basis(bcs) #(NQ, NC, 6, 3)
        cellmeasure = self.mesh.cell_volume()
        cell2dof = self.cell_to_dof()

        if c is None:
            val = np.einsum('q, qckd, qcmd, c->ckm', ws, phi, phi, cellmeasure, optimize=True)
        else:
            if callable(c):
                if c.coordtype == 'barycentric':
                    c = c(bcs)
                elif c.coordtype == 'cartesian':
                    ps = self.mesh.bc_to_point(bcs)
                    c = c(ps)

            if isinstance(c, (int, float)):
                M = np.einsum('q, qcld, qcmd, c->clm', c*ws, phi, phi, cellmeasure, optimize=True)
            else:
                M = np.einsum('q, qcnd, qckn, qcmd, c->ckm', ws, c, phi, phi, cellmeasure, optimize=True)

        I = np.broadcast_to(cell2dof[..., None], M.shape)
        J = np.broadcast_to(cell2dof[:, None, :], M.shape)
        gdof = self.dof.number_of_global_dofs()
        return csr_matrix((M.flat, (I.flat, J.flat)), shape = (gdof, gdof),
                dtype=dtype)

    def face_mass_matrix(self, c = 1, index=np.s_[:]):
        """
        @brief (n \times u, n \times v)_{Gamma_{robin}}
        @param c 系数, 现在只考虑了 c 是常数的情况
        """
        bcs, ws = self.integralalg.faceintegrator.get_quadrature_points_and_weights()
        face2dof = self.dof.face_to_dof(index=index)
        fm = self.mesh.entity_measure("face", index=index)
        fphi = self.face_basis(bcs, index=index)

        EMc = np.einsum('qflg, qfmg, q, f->flm', fphi, fphi, ws, fm)

        gdof = self.dof.number_of_global_dofs()
        I = np.broadcast_to(face2dof[..., None], EMc.shape)
        J = np.broadcast_to(face2dof[:, None, :], EMc.shape)
        EM = c*csr_matrix((EMc.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return EM

    def robin_vector(self, f, isRobinFace):
        """
        @brief 计算 (f, n\times v)_{\Gamma_{robin}} 
        """
        isRobinFace = np.sort(isRobinFace)
        bcs, ws = self.integralalg.faceintegrator.get_quadrature_points_and_weights()
        n = self.mesh.face_unit_normal()[isRobinFace]
        fm = self.mesh.entity_measure("face", index=isRobinFace)

        point = self.mesh.bc_to_point(bcs, index=isRobinFace)
        fval = f(point, n)
        phi = self.face_basis(bcs, index=isRobinFace)

        Fc = np.einsum('qfg, qflg, q, f->fl', fval, phi, ws, fm)

        face2dof = self.dof.face_to_dof(index=isRobinFace)
        gdof = self.dof.number_of_global_dofs()
        F = np.zeros(gdof, dtype=np.float_)
        np.add.at(F, face2dof, Fc)
        return F

    def source_vector(self, f, dtype=np.float_):
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        phi = self.basis(bcs) #(NQ, NC, 6, 3)
        cm = self.mesh.cell_volume()
        cell2dof = self.cell_to_dof()

        point = self.mesh.bc_to_point(bcs) 
        fval = f(point) #(NQ, NC, 3)

        val = np.einsum("qclg, qcg, q, c->cl", phi, fval, ws, cm)
        gdof = self.dof.number_of_global_dofs()
        F = np.zeros(gdof, dtype=dtype)
        np.add.at(F, cell2dof, val)
        return F

    def set_dirichlet_bc(self, gD, uh, threshold=None, q=None):
        """
        """
        mesh = self.mesh
        node = mesh.entity("node")
        edge = mesh.entity("edge")
        face = mesh.entity("face")

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()

        face2edge = mesh.ds.face_to_edge()[index]
        n = mesh.face_unit_normal(index=index) 

        if 0: #节点型自由度
            locEdge = np.array([[1, 2], [2, 0], [0, 1]], dtype=np.int_)
            point = 0.5*(np.sum(node[face[:, locEdge][index]], axis=-2)) #(NF, 3, 3)
            vec = mesh.edge_tangent()[face2edge] #(NF, 3, 3)

            gval = gD(point, n[:, None, :]) #(NF, 3, 3)

            face2dof = self.dof.face_to_dof()[index]
            uh[face2dof] = np.linalg.norm(gval,axis=2) 
        else: #积分型自由度
            bcs, ws = self.integralalg.edgeintegrator.get_quadrature_points_and_weights()
            ps = mesh.bc_to_point(bcs)[:, face2edge]

            vec = mesh.edge_tangent()[face2edge]
            gval = gD(ps, vec)
            l = np.linalg.norm(vec, axis=-1)

            face2dof = self.dof.face_to_dof()[index]
            uh[face2dof] = np.einsum("qfed, fed, q->fe", gval, vec, ws) 

        gdof = self.dof.number_of_global_dofs()
        isDDof = np.zeros(gdof, dtype=np.bool_)
        isDDof[face2dof] = True
        return isDDof

    def set_neumann_bc(self, gN, F=None, threshold=None):
        """

        Notes
        -----
        设置 Neumann 边界条件到载荷向量 F 中

        """
        p = self.p
        mesh = self.mesh
        gdof = self.number_of_global_dofs()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()

        face2dof = self.dof.face_to_dof()[index]

        qf = self.integralalg.faceintegrator 
        bcs, ws = qf.get_quadrature_points_and_weights()

        measure = mesh.entity_measure('face', index=index)

        phi = self.face_basis(bcs)[:, index]
        pp = mesh.bc_to_point(bcs, index=index)
        n = mesh.face_unit_normal(index=index)

        val = gN(pp, n) # (NQ, NF, ...), 这里假设 gN 是一个函数

        if F is None:
            F = np.zeros((gdof, ), dtype=self.ftype)

        bb = np.einsum('q, qfd, qfld, f->fl', ws, val, phi, measure)
        np.add.at(F, face2dof, bb)
        return F

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=dtype)

    def show_basis(self, fig, index=0, box=None):
        """
        Plot quvier graph for every basis in a fig object
        """
        from .femdof import multi_index_matrix3d

        p = self.p
        mesh = self.mesh

        ldof = self.number_of_local_dofs()

        bcs = multi_index_matrix2d(10)/10
        ps = mesh.bc_to_point(bcs)
        phi = self.basis(bcs)

        if p == 0:
            m = 1
            n = 3
        elif p == 1:
            m = 4 
            n = 2 
        elif p == 2:
            m = 5 
            n = 3 
        for i in range(ldof):
            axes = fig.add_subplot(m, n, i+1)
            mesh.add_plot(axes, box=box)
            node = ps[:, index, :]
            uv = phi[:, index, i, :]
            axes.quiver(node[:, 0], node[:, 1], uv[:, 0], uv[:, 1], 
                    units='xy')

