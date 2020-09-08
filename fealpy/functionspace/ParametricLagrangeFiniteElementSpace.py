import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat
from scipy.sparse.linalg import spsolve

from ..decorator import barycentric
from .Function import Function

from ..quadrature import FEMeshIntegralAlg
from ..decorator import timer


class ParametricLagrangeFiniteElementSpace:
    def __init__(self, mesh, p, q=None, spacetype='C'):

        """

        Notes
        -----
            mesh 为一个 Lagrange 网格。

            p 是参数拉格朗日有限元空间的次数，它和 mesh 的次数可以不同。
        """

        self.p = p
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')
        self.dof = mesh.lagrange_dof(p, spacetype=spacetype)
        self.multi_index_matrix = mesh.multi_index_matrix

        self.GD = mesh.geo_dimension()
        self.TD = mesh.top_dimension()

        q = q if q is not None else p+3 
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

        self.itype = mesh.itype
        self.ftype = mesh.ftype

    def __str__(self):
        return "Parametric Lagrange finite element space!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self, index=np.s_[:]):
        return self.dof.cell2dof[index]

    def face_to_dof(self, index=np.s_[:]):
        return self.dof.face_to_dof()

    def edge_to_dof(self, index=np.s_[:]):
        return self.dof.edge_to_dof()

    def is_boundary_dof(self, threshold=None):
        return self.dof.is_boundary_dof(threshold=threshold)

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    @barycentric
    def edge_basis(self, bc):
        phi = self.mesh.shape_function(bc)
        return phi 

    @barycentric
    def face_basis(self, bc):
        phi = self.mesh.shape_function(bc)
        return phi 

    @barycentric
    def basis(self, bc):
        """

        Notes
        -----
        计算基函数在重心坐标点处的函数值，注意 bc 的形状为 (..., TD+1), TD 为 bc
        所在空间的拓扑维数。

        """
        
        p = self.p
        phi = self.mesh.shape_function(bc, p=p)
        return phi 

    @barycentric
    def grad_basis(self, bc, index=np.s_[:], variables='x'):
        """

        Notes
        -----
        计算空间基函数关于实际坐标点 x 的梯度。
        """
        p = self.p
        gphi = self.mesh.grad_shape_function(bc, index=index, p=p,
                variables=variables)
        return gphi

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        phi = self.basis(bc)
        cell2dof = self.dof.cell2dof[index]
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[cell2dof])
        return val

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.dof.cell2dof[index]
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, gphi, uh[cell2dof])
        return val

    def stiff_matrix(self, c=None, q=None, variables='u'):
        if variables == 'x':
            gdof = self.number_of_global_dofs()
            cell2dof = self.cell_to_dof()
            b0 = (self.grad_basis, cell2dof, gdof)
            A = self.integralalg.serial_construct_matrix(b0, c=c, q=q)
            return A 
        elif variables == 'u':
            p = self.p
            qf = self.integralalg.integrator if q is None else self.mesh.integrator(q, etype='cell')
            bcs, ws = qf.get_quadrature_points_and_weights()
            G = self.mesh.first_fundamental_form(bcs)
            gphi = self.mesh.grad_shape_function(bcs, p=p, variables='u')

            rm = self.mesh.reference_cell_measure()
            d = np.sqrt(np.linalg.det(G))
            G = np.linalg.inv(G)
            # c: 单元指标
            # f: 面指标
            # e: 边指标
            # v: 顶点个数指标
            # i, j, k, d: 自由度或基函数指标
            # q: 积分点或重心坐标点指标
            # m, n: 空间或拓扑维数指标
            A = np.einsum('q, qcim, qcmn, qcjn, qc->cij', ws*rm, gphi, G, gphi, d)

            gdof = self.number_of_global_dofs()
            cell2dof = self.cell_to_dof()
            I = np.broadcast_to(cell2dof[:, :, None], shape=A.shape)
            J = np.broadcast_to(cell2dof[:, None, :], shape=A.shape)
            A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
            return A 

    def mass_matrix(self, c=None, q=None):
        """

        Notes
        -----
        组装质量矩阵
        """

        # 积分公式
        qf = self.integralalg.integrator if q is None else self.mesh.integrator(q, etype='cell')
        # bcs : (NQ, n)
        # ws : (NQ, )
        bcs, ws = qf.get_quadrature_points_and_weights() 

        rm = self.mesh.reference_cell_measure() # 参考单元测度
        d = self.mesh.first_fundamental_form(bcs)
        d = np.sqrt(np.linalg.det(d))
        phi = self.mesh.shape_function(bcs, p=self.p)

        # 组装单元矩阵
        M = np.einsum('q, qci, qcj, qc->cij', ws*rm, phi, phi, d) # (NC, ldof, ldof)

        cell2dof = self.cell_to_dof() # (NC, ldof)
        I = np.broadcast_to(cell2dof[:, :, None], shape=M.shape) # (NC, ldof, ldof)
        J = np.broadcast_to(cell2dof[:, None, :], shape=M.shape) # (NC, ldof, ldof)

        # 组装总矩阵
        gdof = self.number_of_global_dofs()
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M 

    def convection_matrix(self, c=None, q=None):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.grad_basis, cell2dof, gdof)
        b1 = (self.basis, cell2dof, gdof)
        A = self.integralalg.serial_construct_matrix(b0, b1=b1, c=c, q=q)
        return A 

    def source_vector(self, f, celltype=False, q=None):
        mesh = self.mesh
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        rm = self.mesh.reference_cell_measure()
        G = self.mesh.first_fundamental_form(bcs)
        d = np.sqrt(np.linalg.det(G))
        ps = mesh.bc_to_point(bcs, etype='cell')
        phi = self.basis(bcs)
        val = f(ps)
        bb = np.einsum('q, qc, qci, qc->ci', ws*rm, val, phi, d)

        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()
        F = np.zeros(gdof, dtype=self.ftype)
        np.add.at(F, cell2dof, bb)
        return F 

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array, coordtype='barycentric')
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=self.ftype)

    def integral_basis(self, q=None):
        """
        """
        cell2dof = self.cell_to_dof()
        qf = self.integrator if q is None else self.mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        rm = self.mesh.reference_cell_measure()
        G = self.mesh.first_fundamental_form(bcs)
        d = np.sqrt(np.linalg.det(G))
        phi = self.basis(bcs)
        cc = np.einsum('q, qci, qc->ci', ws*rm, phi, d)
        gdof = self.number_of_global_dofs()
        c = np.zeros(gdof, dtype=self.ftype)
        np.add.at(c, cell2dof, cc)
        return c

    def interpolation(self, u, dim=None):
        ipoint = self.dof.interpolation_points()
        uI = u(ipoint)
        return self.function(dim=dim, array=uI)

    def set_dirichlet_bc(self, uh, gD, threshold=None, q=None):
        """
        初始化解 uh  的第一类边界条件。
        """

        ipoints = self.interpolation_points()
        isDDof = self.is_boundary_dof(threshold=threshold)
        uh[isDDof] = gD(ipoints[isDDof])
        return isDDof
