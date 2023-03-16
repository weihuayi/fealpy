import argparse
import copy
import numpy as np
from scipy.sparse import csr_matrix, bmat, spdiags
from scipy.sparse.linalg import spsolve

from fealpy.mesh import TriangleMesh, HalfEdgeMesh2d
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.mesh.adaptive_tools import mark

import matplotlib.pyplot as plt

class ContinummDFModel2d:
    def __init__(self, ka=2.7e-3, l0=1.33e-2, la=121.15, mu=80.77):
        """
        @brief

        @param[in] ka 临界能量释放率, 单位kN/mm
        @param[in] l0 尺度系数，单位 mm
        @param[in] la 拉梅第一参数，单位 kN/mm^{-2}
        @param[in] mu 拉梅第二参数, 单位 kN/mm^{-2}
        """

        self.ka = ka
        self.l0 = l0
        self.la = la
        self.mu = mu

        self.index = np.array([
            (0, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 1, 1),
            (0, 2, 0, 0, 0, 1),
#            (1, 0, 1, 1, 0, 0),
            (1, 1, 1, 1, 1, 1),
            (1, 2, 1, 1, 0, 1),
#            (2, 0, 0, 1, 0, 0),
#            (2, 1, 0, 1, 1, 1),
            (2, 2, 0, 1, 0, 1)], dtype=np.int_)

    def init_mesh(self, n=4):
        """
        @brief 生成实始网格
        """
        node = np.array([
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0]], dtype=np.float64)

        cell = np.array([
            [1, 0, 5],
            [4, 5, 0],
            [2, 5, 3],
            [6, 3, 5],
            [4, 7, 5],
            [8, 5, 7],
            [6, 5, 9],
            [8, 9, 5]], dtype=np.int_)

        mesh = TriangleMesh(node, cell)
        mesh.uniform_bisect(n=n)
        mesh = HalfEdgeMesh2d.from_mesh(mesh)
        mesh.ds.NV = 3
        return mesh

    def is_disp_top_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是上边界
        """
        return np.abs(p[..., 1] - 1) < 1e-12

    def is_disp_bottom_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是下边界
        """
        return np.tile(np.abs(p[..., 1]) < 1e-12, 2)

    def strain(self, mesh, uh):
        """
        @brief 给定一个位移，计算相应的应变，这里假设是线性元
        """
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        gphi = mesh.grad_lambda()  # NC x 3 x 2

        s = np.zeros((NC, 2, 2), dtype=np.float64)
        s[:, 0, 0] = np.sum(uh[:, 0][cell] * gphi[:, :, 0], axis=-1)
        s[:, 1, 1] = np.sum(uh[:, 1][cell] * gphi[:, :, 1], axis=-1)

        val = np.sum(uh[:, 0][cell] * gphi[:, :, 1], axis=-1)
        val += np.sum(uh[:, 1][cell] * gphi[:, :, 0], axis=-1)
        val /= 2.0
        s[:, 0, 1] = val
        s[:, 1, 0] = val
        return s

    def stress(self, phi, s):
        """
        @brief 给定应变计算相应的应力
        @param[in] s 单元应变数组，（NC, 2, 2)
        """
        eps = 1e-10

        w, v = self.strain_eigs(s)
        
        # 应变正负分解
        sp, sm = self.strain_pm_eig_decomposition(s)

        ts = np.trace(s, axis1=1, axis2=2)
        tp, tm = self.macaulay_operation(ts)

        la = self.la
        mu = self.mu

        NC = len(s)

        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        c0 = (1 - phi(bc)) ** 2 + eps

        S = np.einsum('i, ijk -> ijk', 2*mu*c0, sp)
        S += 2*mu*sm
        val = c0*la*tp + la*tm
        S[:, 0, 0] += val
        S[:, 1, 1] += val
        return S

    def macaulay_operation(self, alpha):
        """
        @brief 麦考利运算
        """
        val = np.abs(alpha)
        p = (alpha + val) / 2.0
        m = (alpha - val) / 2.0
        return p, m

    def strain_pm_eig_decomposition(self, s):
        """
        @brief 应变的正负特征分解
        @param[in] s 单元应变数组，（NC, 2, 2）
        """
        w, v = np.linalg.eigh(s) # w 特征值, v 特征向量
        p, m = self.macaulay_operation(w)

        sp = np.zeros_like(s)
        sm = np.zeros_like(s)

        for i in range(2):
            n0 = v[:, :, i]  # (NC, 2)
            n1 = p[:, i, None] * n0  # (NC, 2)
            sp += n1[:, :, None] * n0[:, None, :]

            n1 = m[:, i, None] * n0
            sm += n1[:, :, None] * n0[:, None, :]

        return sp, sm

    def strain_energy_density_decomposition(self, s):
        """
        @brief 应变能密度的分解
        """

        la = self.la
        mu = self.mu

        # 应变正负分解
        sp, sm = self.strain_pm_eig_decomposition(s)

        ts = np.trace(s, axis1=1, axis2=2)
        tp, tm = self.macaulay_operation(ts)
        tsp = np.sum(sp ** 2, axis=(1, 2))
        tsm = np.sum(sm ** 2, axis=(1, 2))

        phi_p = la * tp ** 2 / 2.0 + mu * tsp
        phi_m = la * tm ** 2 / 2.0 + mu * tsm
        return phi_p, phi_m

    def strain_eigs(self, s):
        """
        @brief 给定每个单元上的应变，进行特征值分解
        """

        w, v = np.linalg.eig(s)
        return w, v

    def heaviside(self, x, k=1):
        """
        @brief
        """
        val = np.zeros_like(x)
        val[x > 1e-8] = 1
        val[np.abs(x) < 1e-8] = 0.5
        val[x < -1e-8] = 0
        val0 = 1.0 / (1.0 + np.exp(-2 * k * x))
        return val

    def dsigma_depsilon(self, phi, s, bcs):
        """
        @brief 计算应力关于应变的导数矩阵
        @param phi 单元重心处的相场函数值, (NC, )
        @param s 每个单元上的应变矩阵, （NC, 2, 2)
        @return D 单元刚度系数矩阵
        """

        eps = 1e-10
        la = self.la
        mu = self.mu

        NC = len(s)

        D = np.zeros((NC, 3, 3), dtype=np.float64)

        ts = np.trace(s, axis1=1, axis2=2)
        w, v = self.strain_eigs(s)
        hwp = self.heaviside(w)
        hwm = self.heaviside(-w)

        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        c0 = (1 - phi(bc)) ** 2 + eps
#        c0 = (1 - phi) **2 +eps
        c1 = np.zeros_like(c0)
        c2 = np.zeros_like(c0)

        flag = (w[:, 0] == w[:, 1])
        c1[flag] = hwp[flag, 0] / 2.0
        c2[flag] = hwm[flag, 0] / 2.0

        r = np.sum(w[~flag], axis=-1) / np.sum(np.abs(w[~flag]), axis=-1)
        c1[~flag] = (1 + r) / 4.0 
        c2[~flag] = (1 - r) / 4.0


        d0 = 2 * mu * (c0 * hwp[:, 0] + hwm[:, 0])
        d1 = 2 * mu * (c0 * hwp[:, 1] + hwm[:, 1])
        d2 = 2 * mu * (c0 * c1 + c2)

        val = la * (c0 * self.heaviside(ts) + self.heaviside(-ts))
        D[:, 0, 0] = val
        D[:, 0, 1] = val
        D[:, 1, 0] = val
        D[:, 1, 1] = val
        
        for m, n, i, j, k, l in self.index:
            D[:, m, n] += d0 * v[:, i, 0] * v[:, j, 0] * v[:, k, 0] * v[:, l, 0]
            D[:, m, n] += d1 * v[:, i, 1] * v[:, j, 1] * v[:, k, 1] * v[:, l, 1]
            val = v[:, i, 0] * v[:, k, 0] * v[:, j, 1] * v[:, l, 1]
            val += v[:, i, 0] * v[:, l, 0] * v[:, j, 1] * v[:, k, 1]
            val += v[:, i, 1] * v[:, k, 1] * v[:, j, 0] * v[:, l, 0]
            val += v[:, i, 1] * v[:, l, 1] * v[:, j, 0] * v[:, k, 0]
            D[:, m, n] += d2 * val
        D = (D + D.swapaxes(1,2))/2
        return D

    def disp_matrix(self, phi, uh, s):
        NN = mesh.number_of_nodes()
        cellmeasure = mesh.entity_measure('cell')
        space = uh.space
       
        qf = mesh.integrator(4, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad = space.grad_basis(bcs)
        D = self.dsigma_depsilon(phi, s, bcs)
        
        C00 = np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., 0], grad[..., 0], cellmeasure)
        C01 = np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., 0], grad[..., 1], cellmeasure)
        C11 = np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., 1], grad[..., 1], cellmeasure)
        C10 = np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., 1], grad[..., 0], cellmeasure)

        D00 = D[:, 0, 0][:, None, None] * C00
        D00 += D[:, 0, 2][:, None, None] * (C01 + C10)
        D00 += D[:, 2, 2][:, None, None] * C11

        D01 = D[:, 0, 1][:, None, None] * C01
        D01 += D[:, 1, 2][:, None, None] * C11
        D01 += D[:, 0, 2][:, None, None] * C00
        D01 += D[:, 2, 2][:, None, None] * C10

        D10 = D[:, 0, 1][:, None, None] * C10
        D10 += D[:, 1, 2][:, None, None] * C11
        D10 += D[:, 0, 2][:, None, None] * C00
        D10 += D[:, 2, 2][:, None, None] * C01

        D11 = D[:, 1, 1][:, None, None] * C11
        D11 += D[:, 1, 2][:, None, None] * (C01 + C10)
        D11 += D[:, 2, 2][:, None, None] * C00

        cell = mesh.entity('cell')
        shape = D00.shape
        I = np.broadcast_to(cell[:, None, :], shape=shape)
        J = np.broadcast_to(cell[:, :, None], shape=shape)

        D00 = csr_matrix((D00.flat, (I.flat, J.flat)), shape=(NN, NN))
        D01 = csr_matrix((D01.flat, (I.flat, J.flat)), shape=(NN, NN))
        D10 = csr_matrix((D10.flat, (I.flat, J.flat)), shape=(NN, NN))
        D11 = csr_matrix((D11.flat, (I.flat, J.flat)), shape=(NN, NN))

        return bmat([[D00, D01], [D10, D11]], format='csr')

    def phase_matrix(self, mesh, H):

        ka = self.ka
        l0 = self.l0

        mat = np.array([
            [1 / 6, 1 / 12, 1 / 12],
            [1 / 12, 1 / 6, 1 / 12],
            [1 / 12, 1 / 12, 1 / 6]])

        NN = mesh.number_of_nodes()
        cm = mesh.entity_measure('cell')
        gphi = mesh.grad_lambda()  # (NC, 3, 2)

        S = np.einsum('i, ijm, ikm->ijk', ka * l0 * cm, gphi, gphi)
        M = np.einsum('i, jk->ijk', cm * (ka / l0 + 2 * H), mat)

        S += M

        cell = mesh.entity('cell')
        shape = S.shape
        I = np.broadcast_to(cell[:, None, :], shape=shape)
        J = np.broadcast_to(cell[:, :, None], shape=shape)

        A = csr_matrix((S.flat, (I.flat, J.flat)), shape=(NN, NN))

        return A

    def disp_residual(self, S, uh):
        """
        @brief 计算位移右端项
        @param S 每个单元上的应变矩阵, （NC, 2, 2)
        """
        space = uh.space

        cell = mesh.entity('cell')
        NN = mesh.number_of_nodes()
        cm = mesh.entity_measure('cell')
       
        qf = mesh.integrator(4, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs)
        
        bb = np.zeros((cell.shape[0], 3, 2), dtype=np.float64)
        
        b0 = np.einsum('i, j, ijk, j->jk', ws, cm, gphi[..., 0], S[:, 0, 0])
        b1 = np.einsum('i, j, ijk, j->jk', ws, cm, gphi[..., 1], S[:, 0, 1])
        b2 = np.einsum('i, j, ijk, j->jk', ws, cm, gphi[..., 1], S[:, 1, 1])
        b3 = np.einsum('i, j, ijk, j->jk', ws, cm, gphi[..., 0], S[:, 1, 0])
        bb[:, :, 0] = b0 + b1
        bb[:, :, 1] = b2 + b3
        
        b = np.zeros((NN, 2), dtype=np.float64)
        np.add.at(b, cell, bb)
        return -b

    def phase_residual(self, uh, phi, H):
        """
        @brief 计算相场的残量
        """

        ka = self.ka
        l0 = self.l0

        mat = np.array([
            [1 / 6, 1 / 12, 1 / 12],
            [1 / 12, 1 / 6, 1 / 12],
            [1 / 12, 1 / 12, 1 / 6]])

        cell = mesh.entity('cell')

        NN = mesh.number_of_nodes()
        cm = mesh.entity_measure('cell')
        gphi = mesh.grad_lambda()  # (NC, 3, 2)

        M0 = np.einsum('i, jk->ijk', 2 * H * cm, mat)
        M1 = np.einsum('i, jk->ijk', ka / l0 * cm, mat)
        M2 = np.einsum('i, ijm, ikm->ijk', cm * ka * l0, gphi, gphi)

        M = M0 + M1 + M2
        I = np.broadcast_to(cell[:, None, :], shape=M.shape)
        J = np.broadcast_to(cell[:, :, None], shape=M.shape)

        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(NN, NN))

        F = -M @ phi

        bb = H * cm
        bb = bb[:, None] * np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        np.add.at(F, cell, bb)
        return F

def refine_interpolation(mesh, uh_bef, phi_bef):
    nidx = mesh.newnode2node
    halfedge = mesh.entity('halfedge')
    NN_now = mesh.number_of_nodes()
    NN_bef = uh_bef[:].shape[0]
    
    phi_now = np.zeros(NN_now, dtype=np.float_)
    phi_now[:NN_bef] = phi_bef[:]
    phi_now[NN_bef:] = 0.5*(phi_bef[nidx[:, 0]] + phi_bef[nidx[:, 1]])
    
    uh_now = np.zeros((NN_now, uh_bef.shape[-1]), dtype=np.float_)
    uh_now[:NN_bef, :] = uh_bef
    uh_now[NN_bef:, :] = 0.5*(uh_bef[nidx[:, 0], :] + uh_bef[nidx[:, 1], :])
    return uh_now, phi_now


def adaptive_mesh(mesh, d0=0.49, d1=0.75, h=0.005):
    while True:
        cell = mesh.entity("cell")
        node = mesh.entity("node")
        isMarkedCell = mesh.cell_area() > 0.000001
        isMarkedCell = isMarkedCell & (np.min(np.abs(node[cell, 1] - 0.5),
                                              axis=-1) < h)
        isMarkedCell = isMarkedCell & (np.min(node[cell, 0], axis=-1) > d0) & (
                np.min(node[cell, 0], axis=-1) < d1)
        mesh.bisect(isMarkedCell=isMarkedCell)
        if np.all(~isMarkedCell):
            break;

    isMarkedCell = (np.min(np.abs(node[cell, 1] - 0.5),
                           axis=-1) < 0.001)
    isMarkedCell = isMarkedCell & (np.min(node[cell, 0], axis=-1) > d0) & (
            np.min(node[cell, 0], axis=-1) < d1)

    NN = mesh.number_of_nodes()
    cell = mesh.entity("cell")
    edge = mesh.entity('edge')
    node = mesh.entity('node')

    crack = np.zeros(NN, dtype=np.float_)
    crack[cell[isMarkedCell]] = 1.0
    mesh.nodedata['crack'] = crack

    n2cellnum = np.zeros(NN, dtype=np.int_)
    np.add.at(n2cellnum, cell, 1)
    for i in range(5):
        celldata = np.average(crack[cell], axis=-1)
        crack[:] = 0
        np.add.at(crack, cell, celldata[:, None])
        crack = crack / n2cellnum
        crack[cell[isMarkedCell]] = 1.0
    crack[cell[isMarkedCell]] = 1.0
    mesh.nodedata['crack'] = crack

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()
    mesh.to_vtk(fname='ad.vtu')

def recovery_estimate(phi):
    space = phi.space
    rgphi = space.grad_recovery(phi, method='simple')
    eta = space.integralalg.error(rgphi.value, phi.grad_value, power=2,
            celltype=True)
    return eta

## 参数解析
parser = argparse.ArgumentParser(description=
        """
        三角形网格上任意次有限元方法求解连续体断裂模型
        """)

parser.add_argument('--degree',
        default=1, type=int,
        help='Lagrange 有限元空间的次数, 默认为 1 次.')

parser.add_argument('--GD',
        default=2, type=int,
        help='模型问题的维数, 默认求解 2 维问题.')

parser.add_argument('--nrefine',
        default=4, type=int,
        help='初始网格加密的次数, 默认初始加密 4 次.')

parser.add_argument('--maxit',
        default=5, type=int,
        help='默认网格加密次数, 默认加密 5 次')


parser.add_argument('--iteration',
        default=20, type=int,
        help='默认牛顿迭代次数, 默认迭代 20 次')

parser.add_argument('--accuracy',
        default=1e-10, type=float,
        help='默认牛顿迭代精度, 默认精度 1e-10')

args = parser.parse_args()

p = args.degree
GD = args.GD
n = args.nrefine
maxit = args.maxit
iteration = args.iteration
accuracy = args.accuracy

model = ContinummDFModel2d()

mesh = model.init_mesh(n)

space = LagrangeFiniteElementSpace(mesh, p=p)

uh = space.function(dim=GD)
phi = space.function()

NC = mesh.number_of_cells()
H = np.zeros(NC, dtype=np.float64)  # 分片常数
F = space.function(dim=GD)

for i in range(maxit):
    NN = mesh.number_of_nodes()
    node = mesh.entity('node')

    du = np.zeros(NN*2, dtype=np.float64)
    
    isTNode = model.is_disp_top_boundary(node)
    uh[isTNode, 1] += 1e-5

    isTDof = np.r_['0', np.zeros(NN, dtype=np.bool_), isTNode]
    
    isDDof = model.is_disp_bottom_boundary(node)

    k = 0
    while k < iteration:
        print('i:', i)
        print('k:', k)
        
        s = model.strain(mesh, uh)
        
        A = model.disp_matrix(phi, uh, s)
        
        S = model.stress(phi, s)
        F = model.disp_residual(S, uh)
        R0 = F.T.flat
        
        # 边界条件处理
        R0 -= A@du
        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[isDDof] = 1
        bdIdx[isTDof] =1
        Tbd =spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        A = T@A@T + Tbd
        R0[isDDof] = du[isDDof]
        R0[isTDof] = du[isTDof]

        print("求解位移增量")
        du = spsolve(A, R0)
        uh.T.flat += du

        print('uh:', uh)

        s = model.strain(mesh, uh)
        phip, _ = model.strain_energy_density_decomposition(s)

        H1 = np.fmax(H, phip)
        
        NC = mesh.number_of_cells()
        H = np.zeros(NC, dtype=np.float64)  # 分片常数
        H[:] = H1

        R1 = model.phase_residual(uh, phi, H)
        A = model.phase_matrix(mesh, H)

        print("求解相场增量")
        phi += spsolve(A, R1)
        error1 = np.max(np.abs(R0))

        error = max(np.max(np.abs(R0)), np.max(np.abs(R1)))
        print("error1:", error1)
        print("error:", error)
        if error < accuracy:
            break
        k += 1
    eta = recovery_estimate(phi)

    if i < maxit - 1:
        isMarkedCell = mark(eta, theta = 0.2)
        mesh.adaptive_refine(isMarkedCell, method='rg')
        
        space = LagrangeFiniteElementSpace(mesh, p=1)
        uh_new, phi_new = refine_interpolation(mesh, uh, phi)
        uh  = space.function(dim=2) 
        phi = space.function()
        uh[:] = uh_new
        phi[:] = phi_new

#mesh.to_vtk(fname='test.vtu')

fig, axes = plt.subplots()
mesh.node += uh
mesh.add_plot(axes)
plt.show()
