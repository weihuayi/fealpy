import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse import csr_matrix, spdiags, eye, bmat


class DirichletBC():
    """

    Note:

    """
    def __init__(self, space, gD, threshold=None):
        self.space = space
        self.gD = gD
        self.threshold = threshold
        self.bctype = 'Dirichlet'

    def apply(self, A, F, uh=None, threshold=None):
        """

        Notes
        -----

        注意调用这个函数，外界的 F 最后被修改了， 外界的 A 没有修改！
        """
        space = self.space
        gD = self.gD
        threshold = self.threshold if threshold is None else threshold

        gdof = space.number_of_global_dofs()
        GD = A.shape[0]//gdof
        if uh is None:
            uh = self.space.function(dim=GD) # (gdof, GD) 其元素默认为 0 
        isDDof = space.set_dirichlet_bc(gD, uh, threshold=threshold)
        if GD > 1:
            isDDof = np.tile(isDDof, GD)
            F = F.T.flat # (gdof, GD) --> (GD*gdof, ) 把 F 按列展平
        x = uh.T.flat # 把 uh 按列展平
        F -= A@x
        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[isDDof] = 1
        Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd
        F[isDDof] = x[isDDof]
        return A, F 

    def apply_on_matrix(self, A, threshold=None):
        space = self.space
        gdof = space.number_of_global_dofs()
        threshold = self.threshold if threshold is None else threshold

        isDDof = space.boundary_dof(threshold=threshold)
        dim = A.shape[0]//gdof # 如果是向量型问题
        if dim > 1:
            isDDof = np.tile(isDDof, dim)

        bdIdx = np.zeros((A.shape[0], ), np.int_)
        bdIdx[isDDof] = 1
        Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd
        return A

    def apply_on_vector(self, A, F):
        """

        Notes
        -----

        注意调用这个函数，外界的 F 最后被修改了， 外界的 A 没有修改！
        """
        space = self.space
        threshold = self.threshold

        gdof = space.number_of_global_dofs()
        dim = A.shape[0]//gdof
        uh = space.function(dim=dim)
        isDDof = space.set_dirichlet_bc(self.gD, uh, threshold=threshold)
        if dim > 1:
            isDDof = np.tile(isDDof, dim)
            F = F.T.flat
        x = uh.T.flatten() # 把 uh 按列展平
        F -= A@x
        F[isDDof] = x[isDDof] 
        return F 

class NeumannBC():
    def __init__(self, space, gN, threshold=None):
        self.space = space
        self.gN = gN
        self.threshold = threshold
        self.bctype = 'Neumann'

    def apply(self, F, A=None, threshold=None, q=None):
        """

        Parameters
        ----------

        Notes
        -----
            当矩阵 A 不是 None的时候，就假设是纯 Neumann 边界条件，需要同时修改
            矩阵 A 和右端 F, 并返回。

            否则只返回 F

            外界的 F 被修改了

            外界的 A 没有修改
        """
        space = self.space
        gN = self.gN
        threshold = self.threshold if threshold is None else threshold
        F = space.set_neumann_bc(gN, F=F, threshold=threshold, q=q)

        if A is not None: # pure Neumann condtion
            c = space.integral_basis()
            A = bmat([[A, c.reshape(-1, 1)], [c, None]], format='csr')
            F = np.r_[F, 0]
            return A, F
        else:
            return F

class RobinBC():
    def __init__(self, space, gR, threshold=None):
        self.space = space
        self.gR = gR
        self.threshold = threshold
        self.bctype = "Robin"

    def apply(self, A, F, threshold=None, q=None):
        """

        Notes
        -----
            注意调用这个函数，外界的 F 最后被修改了， 外界的 A 没有修改！
        """
        space = self.space
        gR = self.gR
        threshold = self.threshold if threshold is None else threshold
        R, F = space.set_robin_bc(gR, F=F, threshold=threshold, q=q)
        return A+R, F



###
class BoundaryCondition():
    def __init__(self, space, dirichlet=None, neumann=None, robin=None):
        self.space = space
        self.dirichlet = dirichlet
        self.neumann = neumann
        self.robin = robin

    def apply_robin_bc(self, A, b, is_robin_boundary=None):
        """
        apply the Robin boundary condition GD space.

        Parameter
        ---------
        A : matrix with shape (GD*N, GD*N)
        b : vector with shape (GD*N, )

        Returns
        -------

        See also
        --------

        Notes
        -----
        The GD is the dimension of the problem space, and N is the number of
        dofs.

        Examples
        --------

        """
        if self.robin is not None:
            space = self.space
            p = space.p
            mesh = space.mesh
            dim = 1 if len(b.shape) == 1 else b.shape[1]
            face2dof = space.face_to_dof()

            # find the index of all robin boundary 
            idx = mesh.ds.boundary_face_index()
            if is_robin_boundary is not None:
                bc = mesh.entity_barycenter('face', index=idx)
                flag = is_robin_boundary(bc)
                idx = idx[flag]

            measure = mesh.entity_measure('face', index=idx)
            qf = mesh.integrator(p+3, 'face')
            bcs, ws = qf.get_quadrature_points_and_weights()
            phi = space.face_basis(bcs)
            pp = mesh.bc_to_point(bcs, etype='face', index=idx)
            n = mesh.face_unit_normal(index=idx)
            val, kappa = self.robin(pp, n) # (NQ, NF, ...)
            bb = np.einsum('m, mi..., mik, i->ik...', ws, val, phi, measure)
            if dim == 1:
                np.add.at(b, face2dof[idx], bb)
            else:
                np.add.at(b, (face2dof[idx], np.s_[:]), bb)

            FM = np.einsum('m, mi, mij, mik, i->ijk', ws, kappa, phi, phi, measure)

            fdof = space.number_of_local_dofs(etype='face')
            I = np.einsum('k, ij->ijk', np.ones(fdof), face2dof[idx])
            J = I.swapaxes(-1, -2)

            # Construct the stiffness matrix
            A += csr_matrix((FM.flat, (I.flat, J.flat)), shape=A.shape)



    def apply_neumann_bc(self, b, is_neumann_boundary=None):
        """

        Parameters
        ----------
        b : array with shape (N, ) or (N, GD)
        is_neumann_boundary : function object

        Returns
        -------

        See Also
        --------

        Examples
        --------

        """
        if self.neumann is not None:
            space = self.space
            p = space.p
            mesh = space.mesh
            dim = 1 if len(b.shape) == 1 else b.shape[1]
            face2dof = space.face_to_dof()

            # find the index of all neumann boundary 
            idx = mesh.ds.boundary_face_index()
            if is_neumann_boundary is not None:
                bc = mesh.entity_barycenter('face', index=idx)
                flag = is_neumann_boundary(bc)
                idx = idx[flag]
            measure = mesh.entity_measure('face', index=idx)
            qf = mesh.integrator(p+3, 'face')
            bcs, ws = qf.get_quadrature_points_and_weights()
            phi = space.face_basis(bcs)
            pp = mesh.bc_to_point(bcs, etype='face', index=idx)
            n = mesh.face_unit_normal(index=idx)
            val = self.neumann(pp, n) # (NQ, NF, ...)
            bb = np.einsum('m, mi..., mik, i->ik...', ws, val, phi, measure)
            if dim == 1:
                np.add.at(b, face2dof[idx], bb)
            else:
                np.add.at(b, (face2dof[idx], np.s_[:]), bb)


    def apply_dirichlet_bc(self, A, b, uh, is_dirichlet_boundary=None):
        """
        apply the dirichlet boundary condition GD space.

        Parameter
        ---------
        A : matrix with shape (GD*N, GD*N)
        b : vector with shape (GD*N, )
        uh: (N, GD) or (N, ) with GD = 1

        Returns
        -------

        See also
        --------

        Notes
        -----
        The GD is the dimension of the problem space, and N is the number of
        dofs.

        Examples
        --------

        """
        if self.dirichlet is not None:
            isDDof = self.space.set_dirichlet_bc(uh, self.dirichlet,
                    is_dirichlet_boundary)
            dim = 1 if len(uh.shape) == 1 else uh.shape[1]
            if dim > 1:
                isDDof = np.tile(isDDof, dim)
                b = b.T.flat
            gdof = self.space.number_of_global_dofs()
            x = uh.T.flat # 把 uh 按列展平
            b -= A@x
            bdIdx = np.zeros(dim*gdof, dtype=np.int_)
            bdIdx[isDDof] = 1
            Tbd = spdiags(bdIdx, 0, dim*gdof, dim*gdof)
            T = spdiags(1-bdIdx, 0, dim*gdof, dim*gdof)
            A = T@A@T + Tbd
            b[isDDof] = x[isDDof]
            return A, b


