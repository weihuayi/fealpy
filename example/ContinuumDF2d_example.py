
import numpy as np
from scipy.sparse import csr_matrix, bmat
from scipy.sparse.linalg import spsolve

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace

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
            (1, 0, 1, 1, 0, 0),
            (1, 1, 1, 1, 1, 1),
            (1, 2, 1, 1, 0, 1),
            (2, 0, 0, 1, 0, 0),
            (2, 1, 0, 1, 1, 1),
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
        return mesh

    def is_disp_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是上边界
        """
        return np.abs(p[..., 1] - 1) < 1e-12

    def strain(self, uh):
        """
        @brief 给定一个位移，计算相应的应变，这里假设是线性元
        """
        mesh = uh.space.mesh
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        gphi = mesh.grad_lambda() # NC x 3 x 2

        s = np.zeros((NC, 2, 2), dtype=np.float64)
        s[:, 0, 0] = np.sum(uh[:, 0][cell]*gphi[:, :, 0], axis=-1) 
        s[:, 1, 1] = np.sum(uh[:, 1][cell]*gphi[:, :, 1], axis=-1)

        val = np.sum(uh[:, 0][cell]*gphi[:, :, 1], axis=-1)
        val += np.sum(uh[:, 1][cell]*gphi[:, :, 0], axis=-1)
        val /=2.0 
        s[:, 0, 1] = val
        s[:, 1, 0] = val 
        return s

    def stress(self, s):
        """
        @brief 给定应变计算相应的应力

        @param[in] s 单元应变数组，（NC, 2, 2)
        """

        la = self.la
        mu = self.mu

        mesh = uh.space.mesh
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        gphi = mesh.grad_lambda() # NC x 3 x 2

        S = np.zeros((NC, 2, 2), dtype=np.float64)

        S[:, 0, 0] = (2*mu + la)*s[:, 0, 0] + la*s[:, 1, 1]
        S[:, 1, 1] = la*s[:, 0, 0] + (2*mu + la)*s[:, 1, 1]
        S[:, 0, 1] = 2*mu*s[:, 0, 1]
        S[:, 1, 0] = S[:, 0, 1]
        return S

    def macaulay_operation(self, alpha):
        """
        @brief 麦考利运算
        """
        val = np.abs(alpha)
        p = (alpha + val)/2.0
        m = (alpha - val)/2.0
        return p, m 
        

    def strain_pm_eig_decomposition(self, s):
        """
        @brief 应变的正负特征分解

        @param[in] s 单元应变数组，（NC, 2, 2）
        """
        w, v = np.linalg.eigh(s)
        p, m = self.macaulay_operation(w)

        sp = np.zeros_like(s)
        sm = np.zeros_like(s)

        for i in range(2):
            n0 = v[:, :, i] # (NC, 2)
            n1 = p[:, i, None]*n0 # (NC, 2)
            sp += n1[:, :, None]*n0[:, None, :]
            
            n1 = m[:, i, None]*n0
            sm += n1[:, :, None]*n0[:, None, :]

        return sp, sm


    def strain_energy_density_decomposition(self, s):
        """
        @brief 应变能密度的分解
        """

        la = self.la
        mu = self.mu

        sp, sm = self.strain_pm_eig_decomposition(s)

        ts = np.trace(s, axis1=1, axis2=2)
        tp, tm = self.macaulay_operation(ts)
        tsp = np.sum(sp**2, axis=(1, 2)) 
        tsm = np.sum(sm**2, axis=(1, 2))

        phi_p = la*tp**2/2.0 + mu*tsp
        phi_m = la*tm**2/2.0 + mu*tsm
        return phi_p, phi_m


    def strain_eigs(self, s):
        """
        @brief 给定每个单元上的应变，进行特征值分解
        """

        w, v = np.linalg.eigh(s)
        return w, v

    def heaviside(self, x, k=1):
        """
        @brief 
        """
        val = 1.0/(1.0 + np.exp(-2*k*x))
        return val


    def dsigma_depsilon(self, phi, s):
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

        bc = np.array([1/3, 1/3, 1/3], dtype=np.float64)
        c0 = (1 - phi(bc))**2 + eps

        c1 = np.zeros_like(c0)
        c2 = np.zeros_like(c0)

        flag = (w[:, 0] == w[:, 1])
        c1[flag] = hwp[flag, 0]
        c2[flag] = hwm[flag, 0]

        r = np.sum(w[~flag], axis=-1)/np.sum(np.abs(w[~flag]), axis=-1)
        c1[~flag] = (1 + r)/4.0 
        c2[~flag] = (1 - r)/4.0

        d0 = 2*mu*(c0*hwp[:, 0] + hwm[:, 0])
        d1 = 2*mu*(c0*hwp[:, 1] + hwm[:, 1])
        d2 = 2*mu*(c0*c1 + c2)

        val = la*(c0*self.heaviside(ts) + self.heaviside(-ts))
        D[:, 0, 0] = val
        D[:, 0, 1] = val
        D[:, 1, 0] = val
        D[:, 1, 1] = val

        for m, n, i, j, k, l in self.index:
            D[:, m, n] += d0*v[:, i, 0]*v[:, j, 0]*v[:, k, 0]*v[:, l, 0]  
            D[:, m, n] += d1*v[:, i, 1]*v[:, j, 1]*v[:, k, 1]*v[:, l, 1]
            val  = v[:, i, 0]*v[:, k, 0]*v[:, j, 1]*v[:, l, 1]
            val += v[:, i, 0]*v[:, l, 0]*v[:, j, 1]*v[:, k, 1]
            val += v[:, i, 1]*v[:, k, 1]*v[:, j, 0]*v[:, l, 0]
            val += v[:, i, 1]*v[:, l, 1]*v[:, j, 0]*v[:, k, 0]
            D[:, m, n] += d2*val

        return D

    def disp_matrix(self, mesh, D):

        NN = mesh.number_of_cells()
        cm = mesh.entity_measure('cell')
        gphi = mesh.grad_lambda() # (NC, 3, 2)

        C00 = np.einsum('ij, ik->ijk', gphi[:, :, 0], gphi[:, :, 0]) # (NC, 3, 3)
        C11 = np.einsum('ij, ik->ijk', gphi[:, :, 1], gphi[:, :, 1]) # (NC, 3, 3)
        C01 = np.einsum('ij, ik->ijk', gphi[:, :, 0], gphi[:, :, 1]) # (NC, 3, 3)
        C10 = np.einsum('ij, ik->ijk', gphi[:, :, 1], gphi[:, :, 0])


        D00  = D[:, 0, 0][:, None, None]*C00 
        D00 += D[:, 0, 2][:, None, None]*(C01 + C10)
        D00 += D[:, 2, 2][:, None, None]*C11

        D00 *= cm[:, None, None]

        D01  = D[:, 0, 1][:, None, None]*C01
        D01 += D[:, 1, 2][:, None, None]*C11
        D01 += D[:, 0, 2][:, None, None]*C00
        D01 += D[:, 2, 2][:, None, None]*C10

        D01 *= cm[:, None, None]


        D11  = D[:, 1, 1][:, None, None]*C11
        D11 += D[:, 1, 2][:, None, None]*(C01 + C10)
        D11 += D[:, 2, 2][:, None, None]*C00

        D11 *= cm[:, None, None]

        cell = mesh.entity('cell')
        shape = D00.shape
        I = np.broadcast_to(cell[:, None, :], shape=shape)
        J = np.broadcast_to(cell[:, :, None], shape=shape)

        D00 = csr_matrix((D00.flat, (I.flat, J.flat)), shape=(NN, NN))
        D01 = csr_matrix((D01.flat, (I.flat, J.flat)), shape=(NN, NN))
        D11 = csr_matrix((D11.flat, (I.flat, J.flat)), shape=(NN, NN))

        return bmat([[D00, D01], [D01.T, D11]], format='csr')


    def phase_matrix(self, mesh, H):

        ka = self.ka
        l0 = self.l0

        mat = np.array([
            [1/6, 1/12, 1/12], 
            [1/12, 1/6, 1/12], 
            [1/12, 1/12, 1/6]]) 

        NN = mesh.number_of_nodes()
        cm = mesh.entity_measure('cell')
        gphi = mesh.grad_lambda() # (NC, 3, 2)

        S = np.einsum('i, ijm, ikm->ijk', ka*l0*cm, gphi, gphi)
        M = np.einsum('i, jk->ijk', cm*(ka/l0 + 2*H), mat)

        S += M

        cell = mesh.entity('cell')
        shape = S.shape
        I = np.broadcast_to(cell[:, None, :], shape=shape)
        J = np.broadcast_to(cell[:, :, None], shape=shape)

        A = csr_matrix((S.flat, (I.flat, J.flat)), shape=(NN, NN))

        return A


    def disp_residual(self, uh, phi, K):
        return -K@uh.T.flat


    def phase_residual(self, uh, phi, H):
        """
        @brief 计算相场的残量 
        """

        ka = self.ka
        l0 = self.l0

        mat = np.array([
            [1/6, 1/12, 1/12], 
            [1/12, 1/6, 1/12], 
            [1/12, 1/12, 1/6]]) 

        cell = mesh.entity('cell')

        NN = mesh.number_of_nodes()
        cm = mesh.entity_measure('cell')
        gphi = mesh.grad_lambda() # (NC, 3, 2)

        M0 = np.einsum('i, jk->ijk', 2*H*cm, mat)
        M1 = np.einsum('i, jk->ijk', ka/l0*cm, mat) 
        M2 = np.einsum('i, ijm, ikm->ijk', cm*ka*l0, gphi, gphi)

        M = M0 + M1 + M2
        I = np.broadcast_to(cell[:, None, :], shape=M.shape)
        J = np.broadcast_to(cell[:, :, None], shape=M.shape)

        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(NN, NN))

        F = -M@phi

        bb = H*cm
        bb = bb[:, None]*np.array([1/3, 1/3, 1/3], dtype=np.float64)
        np.add.at(F, cell, bb)

        return F

def adaptive_mesh(mesh, d0 = 0.49, d1 = 0.75, h=0.005):
    while True:
        cell = mesh.entity("cell")
        node = mesh.entity("node")
        isMarkedCell = mesh.cell_area()>0.000001
        isMarkedCell = isMarkedCell & (np.min(np.abs(node[cell, 1]-0.5),
            axis=-1) < h)
        isMarkedCell = isMarkedCell & (np.min(node[cell, 0], axis=-1)>d0) &(
                np.min(node[cell, 0], axis=-1)<d1)
        mesh.bisect(isMarkedCell=isMarkedCell)
        if np.all(~isMarkedCell):
            break;

    isMarkedCell = (np.min(np.abs(node[cell, 1]-0.5),
        axis=-1) < 0.001)
    isMarkedCell = isMarkedCell & (np.min(node[cell, 0], axis=-1)>d0) &(
        np.min(node[cell, 0], axis=-1)<d1)

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
        crack = crack/n2cellnum
        crack[cell[isMarkedCell]] = 1.0
    crack[cell[isMarkedCell]] = 1.0
    mesh.nodedata['crack'] = crack

    fig = plt.figure()
    axes = fig.gca()
    mesh.add_plot(axes)
    plt.show()
    mesh.to_vtk(fname='ad.vtu')


p = 1
n = 0
maxit = 10

model = ContinummDFModel2d()

mesh = model.init_mesh(4)
NC = mesh.number_of_cells()
NN = mesh.number_of_nodes()
node = mesh.entity('node')


isBdNode = model.is_disp_boundary(node)

isInDof = np.r_['0', np.ones(NN, dtype=np.bool_), ~isBdNode]


space = LagrangeFiniteElementSpace(mesh, p=p)

K = space.linear_elasticity_matrix(model.la, model.mu, q=1) # 线弹性刚度矩阵

uh = space.function(dim=2)
phi = space.function()
H = np.zeros(NC, dtype=np.float64) # 分片常数 

for i in range(1):
    uh[isBdNode, 1] += 1e-5
    
    k=0
    while k < maxit:
        R0 = model.disp_residual(uh, phi, K)
        
        s = model.strain(uh)
        D = model.dsigma_depsilon(phi, s)
        print(D)
        A = model.disp_matrix(mesh, D)

        print("求解位移增量")
        du = spsolve(A[isInDof, :][:, isInDof], R0[isInDof])

        uh.T.flat[isInDof] += du

        s = model.strain(uh)
        phip, _ = model.strain_energy_density_decomposition(s)

        H = np.fmax(H, phip)

        R1 = model.phase_residual(uh, phi, H)
        A = model.phase_matrix(mesh, H)

        print("求解相场增量")
        phi += spsolve(A, R1)

        error = max(np.max(np.abs(R0)), np.max(np.abs(R1))) 
        print(error)
        if  error < 1e-5:
            break

        k += 1
        

fig, axes = plt.subplots()
mesh.add_plot(axes)
plt.show()
adaptive_mesh(mesh, d0 = 0.499, d1=1, h=0.001)
#fig, axes = plt.subplots()
#mesh.add_plot(axes)
#plt.show()

