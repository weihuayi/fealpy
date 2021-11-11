'''
Title: 基于Fealpy的电磁反向波有限元仿真

Author:  王唯

E-mail: <abelwangwei@163.com>

Address: 湘潭大学  数学与计算科学学院

'''

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt


from fealpy.mesh import TriangleMesh
from fealpy.mesh import MeshFactory as mf
from fealpy.functionspace import FirstKindNedelecFiniteElementSpace2d
from fealpy.boundarycondition import DirichletBC
from fealpy.boundarycondition import BoundaryCondition

from fealpy.decorator import cartesian, barycentric
from fealpy.tools.show import showmultirate, show_error_table
import matplotlib.quiver
import matplotlib as mpl
import sympy


class PML_model:
    def __init__(self, kappa=None, absortion_constant=None, pml_delta_x=None, pml_delta_y=None):
        self.kappa = 10  # 波数
        self.absortion_constant = 100  # 吸收常数
        self.pml_delta_x = 0.5  # x-PML的厚度
        self.pml_delta_y = 0.5  # y-PML的厚度

    def domain(self):
        val = np.array([-2, 2, -5, 2])
        return val

    def pml_sigma_x(self, val_x):
        domain = self.domain()
        d_x = self.pml_delta_x
        C = self.absortion_constant

        a1 = domain[0] + d_x
        b1 = domain[1] - d_x

        x = val_x
        val = np.zeros_like(x)

        idx_1 = (x > domain[0]) & (x < a1)
        idx_2 = (x > a1) & (x < b1)
        idx_3 = (x > b1) & (x < domain[1])

        val[idx_1] = C * (((x[idx_1] - a1) / d_x) ** 2)
        val[idx_2] = 0.0
        val[idx_3] = C * (((x[idx_3] - b1) / d_x) ** 2)

        return val

    def pml_sigma_y(self, val_y):
        domain = self.domain()

        d_y = self.pml_delta_y
        C = self.absortion_constant

        a2 = domain[2] + d_y
        b2 = domain[3] - d_y

        y = val_y
        val = np.zeros_like(val_y)

        idx_1 = (y > domain[2]) & (y < a2)
        idx_2 = (y > a2) & (y < b2)
        idx_3 = (y > b2) & (y < domain[3])

        val[idx_1] = C * (((y[idx_1] - a2) / d_y) ** 2)
        val[idx_2] = 0.0
        val[idx_3] = C * (((y[idx_3] - b2) / d_y) ** 2)

        return val

    def pml_d_x(self, point):
        sigma = self.pml_sigma_x(point)
        k = self.kappa
        val = 1.0 + (sigma / (2.0 * np.pi * k)) * 1j
        return val

    def pml_d_y(self, point):
        sigma = self.pml_sigma_y(point)
        k = self.kappa
        val = 1.0 + (sigma / (2.0 * np.pi * k)) * 1j
        return val

    @cartesian
    def pml_alpha(self, p):
        x = p[..., 0]
        y = p[..., 1]

        pml_d_x = self.pml_d_x(point=x)
        pml_d_y = self.pml_d_y(point=y)
        val = (pml_d_x * pml_d_y)

        return val

    @cartesian
    def pml_beta(self, p):
        x = p[..., 0]
        y = p[..., 1]
        w = np.size(p, 0)
        t = np.size(p, 1)

        val = np.zeros([w, t, 2, 2], dtype=np.complex_)
        pml_d_x = self.pml_d_x(point=x)
        pml_d_y = self.pml_d_y(point=y)

        # 这个地方要求旋度
        val[..., 0, 0] = 1.0 / pml_d_y ** 2
        val[..., 1, 1] = 1.0 / pml_d_x ** 2

        return val

    @cartesian
    def epsilon(self, p):
        x = p[..., 0]
        y = p[..., 1]
        idx = (x > -2) & (x < 2) & (y > -2.4) & (y < -0.6)

        w = np.size(p, 0)
        t = np.size(p, 1)
        val = np.ones([w, t, 1])
        val[idx, 0] = 1.0 / - 1.1
        return val

    @cartesian
    def mu(self, p):
        x = p[..., 0]
        y = p[..., 1]
        idx = (x > -2) & (x < 2) & (y > -2.4) & (y < -0.6)

        w = np.size(p, 0)
        t = np.size(p, 1)
        val = np.zeros([w, t, 2, 2])
        val[..., 0, 0] = 1.0
        val[..., 1, 1] = 1.0
        val[idx, 0, 0] = -1.1
        val[idx, 1, 1] = -1.1
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        k = self.kappa
        idx = (p[..., 0] ** 2 + (p[..., 1] - 1.45) ** 2 < 0.5 ** 2)
        val = np.zeros(p.shape, dtype=np.complex_)
        val[idx, 0] = (k ** 2) * (1 - 4 * ((x[idx] ** 2) + ((y[idx] - 1.45) ** 2))) * np.exp(1.j * k * y[idx])
        val[..., 1] = 0.0
        return val

    @cartesian
    def dirichlet(self, p, t):
        no = np.zeros_like(p)
        no += 1e-14
        val = np.sum(no, axis=-1)
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        eps = 1e-14
        return (x >= -2 - eps) | (x <= 2 + eps) | (y >= -5 - eps) | (y <= 2 + eps)


class PML_solution:
    def __init__(self, p=None, q=None, n=None, kappa=None, pde=None, mesh=None, space=None):
        self.p = 0  # 有限元空间次数=p+1
        self.q = 9  # 积分精度
        self.n = 100  # 网格划分尺寸
        self.pde = PML_model()
        self.kappa = self.pde.kappa
        self.mesh = mf.boxmesh2d(self.pde.domain(), nx=self.n, ny=self.n, meshtype='tri')
        self.space = FirstKindNedelecFiniteElementSpace2d(mesh=self.mesh, p=self.p)

    def get_qf_bcs_ws_ps(self):
        # 得到每个单元上的重心坐标, 高斯积分点, 高斯积分权重
        qf = self.mesh.integrator(self.q, etype='cell')

        # 重心坐标, 权重
        bcs, ws = qf.get_quadrature_points_and_weights()

        # 得到自然坐标
        ps = self.mesh.bc_to_point(bcs)  # (积分点个数, 单元个数, 坐标轴个数)
        return qf, bcs, ws, ps

    def cellmeasure(self):
        val = self.mesh.entity_measure('cell')
        return val

    def get_matching_alpha(self):
        qf, bcs, ws, ps = self.get_qf_bcs_ws_ps()
        alpha = self.pde.pml_alpha(ps)
        return alpha

    def get_matching_beta(self):
        qf, bcs, ws, ps = self.get_qf_bcs_ws_ps()
        beta = self.pde.pml_beta(ps)
        return beta

    def curl_phi(self):
        space = self.space
        qf, bcs, ws, ps = self.get_qf_bcs_ws_ps()
        curl_phi = space.curl_basis(bcs)
        return curl_phi

    def phi(self):
        space = self.space
        qf, bcs, ws, ps = self.get_qf_bcs_ws_ps()
        phi = space.basis(bcs)
        return phi

    def get_curl_matrix(self):
        qf, bcs, ws, ps = self.get_qf_bcs_ws_ps()
        cellmeasure = self.cellmeasure()

        alpha = self.get_matching_alpha()
        eps = self.pde.epsilon(ps)
        mu = self.pde.mu(ps)

        curl_phi = self.curl_phi()

        curl_M = np.einsum('i, ijq, ijss, ijss, ij, ijk, ijm, j->jkm', ws, eps, mu, mu, alpha, curl_phi, curl_phi,
                           cellmeasure)

        cell2dof = self.space.cell_to_dof()
        # # #
        I = np.broadcast_to(cell2dof[:, :, None], shape=curl_M.shape)
        #
        J = np.broadcast_to(cell2dof[:, None, :], shape=curl_M.shape)
        #
        gdof = self.space.number_of_global_dofs()
        #
        curl_matrix = csr_matrix((curl_M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return curl_matrix

    def get_mass_matrix(self):
        qf, bcs, ws, ps = self.get_qf_bcs_ws_ps()
        cellmeasure = self.cellmeasure()
        beta = self.get_matching_beta()
        phi = self.phi()
        mu = self.pde.mu(ps)
        k = self.pde.kappa

        M2 = np.einsum('i, ijdd, ijdd, ijkd, ijmd, j->jkm', ws * (k ** 2), mu, beta, phi, phi, cellmeasure)

        cell2dof = self.space.cell_to_dof()
        #
        I = np.broadcast_to(cell2dof[:, :, None], shape=M2.shape)
        #
        J = np.broadcast_to(cell2dof[:, None, :], shape=M2.shape)
        #
        gdof = self.space.number_of_global_dofs()
        #
        mass_matrix = csr_matrix((M2.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return mass_matrix

    def get_right(self):
        qf, bcs, ws, ps = self.get_qf_bcs_ws_ps()
        cellmeasure = self.cellmeasure()
        beta = self.get_matching_beta()
        phi = self.phi()
        mu = self.pde.mu(ps)
        val = self.pde.source(ps)

        cell2dof = self.space.cell_to_dof()

        bb = np.einsum('i, ijss, ijm, ijmq, ijkm, j->jk', ws, mu, val, beta, phi, cellmeasure)

        gdof = self.space.number_of_global_dofs()

        F = np.zeros(gdof, dtype=np.complex_)

        np.add.at(F, cell2dof, bb)

        return F

    def get_left(self):
        k = self.kappa
        A = self.get_curl_matrix()
        B = self.get_mass_matrix()
        val = A - B
        return val

    def dual_boundary(self):
        A = self.get_left()
        F = self.get_right()
        uh = self.space.function(dtype=np.complex_)
        bc = DirichletBC(self.space, self.pde.dirichlet, threshold=self.pde.is_dirichlet_boundary)
        A, F = bc.apply(A, F, uh)
        return A, F, uh

    def algebra_system(self):
        A, F, uh = self.dual_boundary()
        uh[:] = spsolve(A, F)
        return uh

    def get_imag(self):
        mesh = self.mesh
        uh = self.algebra_system()
        aa = np.array([1 / 3, 1 / 3, 1 / 3])
        pp = mesh.bc_to_point(aa)
        value = uh(aa)
        mesh.add_plot(plt, cellcolor=value[..., 0].imag, linewidths=0)
        plt.show()


pml = PML_solution()
pml.get_imag()
