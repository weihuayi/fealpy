import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, bmat, spdiags

from fealpy.mesh import TriangleMesh 
from fealpy.geometry import SquareWithCircleHoleDomain

from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm

from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarMassIntegrator
from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import ProvidesSymmetricTangentOperatorIntegrator

from fealpy.fem import DirichletBC
from fealpy.fem import recovery_alg
from fealpy.mesh.adaptive_tools import mark


class Brittle_Facture_model():
    def __init__(self):
        self.E = 200 # 杨氏模量
        self.nu = 0.2 # 泊松比
        self.Gc = 1 # 材料的临界能量释放率
        self.l0 = 0.02 # 尺度参数，断裂裂纹的宽度

        self.mu = self.E / (1 + self.nu) / 2.0 # 剪切模量
        self.lam = self.E * self.nu / (1 + self.nu) / (1- 2*self.nu) # 拉梅常数
        self.kappa = self.lam + 2 * self.mu /3 # 压缩模量

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
        mesh.ds.NV = 3
        return mesh

    def top_boundary_disp(self):
        """
        @brief Dirichlet 边界条件，位移边界条件
        Notes
        -----
        这里向量的第 i 个值表示第 i 个时间步的位移的大小
        """
        return np.concatenate((np.linspace(0, 70e-3, 6)[1:], np.linspace(70e-3,
            125e-3, 26)[1:]))

    def top_disp_direction(self):
        """
        @brief 上边界位移的方向
        Notes
        -----
        位移方向沿 (0, 1) 方向，即仅在 y 方向的位移变化
        """
        return np.array([0, 1], np.float_)

    def is_top_boundary(self, p):
        """
        @brief 标记上边界, y = 1 时的边界点
        """
        return np.abs(p[..., 1] - 1) < 1e-12 

    def is_inter_boundary(self, p):
        """
        @brief 标记内部边界, 内部圆的点
        Notes
        -----
        内部圆周的点为 DirichletBC，相场值和位移均为 0
        """
        return np.abs((p[..., 0]-0.5)**2 + np.abs(p[..., 1]-0.5)**2 - 0.04) < 0.001
    
    def is_below_boundary(self, p):
        """
        @brief 标记位移加载边界条件，该模型是下边界
        """
        return np.abs(p[..., 1]) < 1e-12

class fracture_damage_integrator():
    def __init__(self, mesh, model):
        self.la = model.lam
        self.mu = model.mu
        self.ka = model.Gc
        self.l0 = model.l0
        
        self.mesh = mesh
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

    def strain(self, uh):
        """
        @brief 给定一个位移，计算相应的应变，这里假设是线性元
        """
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        gphi = mesh.grad_lambda()  # NC x 3 x 2

        s = np.zeros((NC, 2, 2), dtype=np.float64)
        uh = uh.T
        s[:, 0, 0] = np.sum(uh[:, 0][cell] * gphi[:, :, 0], axis=-1)
        s[:, 1, 1] = np.sum(uh[:, 1][cell] * gphi[:, :, 1], axis=-1)

        val = np.sum(uh[:, 0][cell] * gphi[:, :, 1], axis=-1)
        val += np.sum(uh[:, 1][cell] * gphi[:, :, 0], axis=-1)
        val /= 2.0
        s[:, 0, 1] = val
        s[:, 1, 0] = val
        return s
    
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
        tsp = np.trace(sp**2, axis1=1, axis2=2)
        tsm = np.trace(sm**2, axis1=1, axis2=2)

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
        val[x > 1e-13] = 1
        val[np.abs(x) < 1e-13] = 0.5
        val[x < -1e-13] = 0
        return val

    def dsigma_depsilon(self, phi, uh):
        """
        @brief 计算应力关于应变的导数矩阵
        @param phi 单元重心处的相场函数值, (NC, )
        @param uh 位移
        @return D 单元刚度系数矩阵
        """

        eps = 1e-10
        la = self.la
        mu = self.mu
        s = self.strain(uh)

        NC = len(s)
        D = np.zeros((NC, 3, 3), dtype=np.float64)

        ts = np.trace(s, axis1=1, axis2=2)
        w, v = self.strain_eigs(s)
        hwp = self.heaviside(w)
        hwm = self.heaviside(-w)

        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        c0 = (1 - phi(bc)) ** 2 + eps
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
    
    def get_dissipated_energy(self, d):

        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        mesh = self.mesh
        cm = mesh.entity_measure('cell')
        g = d.grad_value(bc)

        val = self.ka/2/self.l0*(d(bc)**2+self.l0**2*np.sum(g*g, axis=1))
        dissipated = np.dot(val, cm)
        return dissipated

    
    def get_stored_energy(self, psi_s, d):
        eps = 1e-10

        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        c0 = (1 - d(bc)) ** 2 + eps
        mesh = self.mesh
        cm = mesh.entity_measure('cell')
        val = c0*psi_s
        stored = np.dot(val, cm)
        return stored


def recovery_estimate(mesh, d):
    from fealpy.functionspace import LagrangeFiniteElementSpace
    space0 = LagrangeFiniteElementSpace(mesh)
    rgd = space0.grad_recovery(uh=d, method='simple')
    eta = space0.integralalg.error(rgd.value, d.grad_value, power=2,
            celltype=True)
    return eta

model = Brittle_Facture_model()

domain = SquareWithCircleHoleDomain() 
mesh = TriangleMesh.from_domain_distmesh(domain, 0.02, maxit=100)
#mesh = model.init_mesh(n=5)

GD = mesh.geo_dimension()
NC = mesh.number_of_cells()
NN = mesh.number_of_nodes()

simulation = fracture_damage_integrator(mesh, model)
space = LagrangeFESpace(mesh, p=1, doforder='sdofs')

d = space.function()
H = np.zeros(NC, dtype=np.float64)  # 分片常数
uh = space.function(dim=GD)
du = space.function(dim=GD)
disp = model.top_boundary_disp()

stored_energy = np.zeros(len(disp)+1, dtype=np.float64)
dissipated_energy = np.zeros(len(disp)+1, dtype=np.float64)

for i in range(len(disp)):
    node  = mesh.entity('node') 
    isTNode = model.is_top_boundary(node)
    uh[1, isTNode] = disp[i]
    isTDof = np.r_['0', np.zeros(NN, dtype=np.bool_), isTNode]

    k = 0
    while k < 100:
        print('i:', i)
        print('k:', k)
        
        # 求解位移
        vspace = (GD*(space, ))
        ubform = BilinearForm(GD*(space, ))

        D = simulation.dsigma_depsilon(d, uh)
        integrator = ProvidesSymmetricTangentOperatorIntegrator(D, q=4)
        ubform.add_domain_integrator(integrator)
        ubform.assembly()
        A0 = ubform.get_matrix()
        R0 = -A0@uh.flat[:]
        
        ubc = DirichletBC(vspace, 0, threshold=model.is_inter_boundary)
        A0, R0 = ubc.apply(A0, R0)
        
        # 位移边界条件处理
        bdIdx = np.zeros(A0.shape[0], dtype=np.int_)
        bdIdx[isTDof] =1
        Tbd =spdiags(bdIdx, 0, A0.shape[0], A0.shape[0])
        T = spdiags(1-bdIdx, 0, A0.shape[0], A0.shape[0])
        A0 = T@A0@T + Tbd
        R0[isTDof] = du.flat[isTDof]
        
        du.flat[:] = spsolve(A0, R0)
        uh[:] += du
        
        # 更新参数
        strain = simulation.strain(uh)
        phip, _ = simulation.strain_energy_density_decomposition(strain)
        H[:] = np.fmax(H, phip)

        # 计算相场模型
        dbform = BilinearForm(space)
        dbform.add_domain_integrator(ScalarDiffusionIntegrator(c=model.Gc*model.l0,
            q=4))
        dbform.add_domain_integrator(ScalarMassIntegrator(c=2*H+model.Gc/model.l0, q=4))
#        dbform.add_domain_integrator(ScalarMassIntegrator(c=model.Gc/model.l0))
        dbform.assembly()
        A1 = dbform.get_matrix()

        lform = LinearForm(space)
        lform.add_domain_integrator(ScalarSourceIntegrator(2*H, q=4))
        lform.assembly()
        R1 = lform.get_vector()
        R1 -= A1@d[:]
        dbc = DirichletBC(space, 0, threshold=model.is_inter_boundary)
        A1, R1 = dbc.apply(A1, R1)
        d[:] += spsolve(A1, R1)
        
        # 计算残量误差
        if k == 0:
            er0 = np.linalg.norm(R0)
            er1 = np.linalg.norm(R1)
        error0 = np.linalg.norm(R0)/er0
        print("error0:", error0)

        error1 = np.linalg.norm(R1)/er1
        print("error1:", error1)
        error = max(error0, error1)
        print("error:", error)
        if error < 1e-5:
            break
        k += 1
    stored_energy[i+1] = simulation.get_stored_energy(phip, d)
    dissipated_energy[i+1] = simulation.get_dissipated_energy(d)

    mesh.nodedata['damage'] = d
    mesh.nodedata['uh'] = uh.T
    fname = 'test' + str(i).zfill(10)  + '.vtu'
    mesh.to_vtk(fname=fname)
    if i < len(disp) - 1:
        cell2dof = mesh.cell_to_ipoint(p=1)
        uh0c2f = uh[0, cell2dof]
        uh1c2f = uh[1, cell2dof]
        dc2f = d[cell2dof]
        data = {'uh0':uh0c2f, 'uh1':uh1c2f, 'd':dc2f, 'H':H[cell2dof]}

        recovery = recovery_alg(space)
        eta = recovery.recovery_estimate(d)
#        option = mesh.adaptive_options(data=data, disp=False)
#        mesh.adaptive(eta, options=option)

        isMarkedCell = mark(eta, theta = 0.2)
        option = mesh.bisect_options(data=data, disp=False)
        mesh.bisect(isMarkedCell, options=option)
       
        space = LagrangeFESpace(mesh, p=1, doforder='sdofs')
        cell2dof = space.cell_to_dof()
        H[cell2dof.reshape(-1)] = option['data']['H'].reshape(-1)
        uh[0, cell2dof.reshape(-1)] = option['data']['uh0'].reshape(-1)
        uh[1, cell2dof.reshape(-1)] = option['data']['uh1'].reshape(-1)
        d[cell2dof.reshape(-1)] = option['data']['d'].reshape(-1)

fig = plt.figure()
axes = fig.add_subplot(111)
mesh.node += uh[:, :NN].T
mesh.add_plot(axes)
plt.show()

plt.figure()
plt.plot(disp, stored_energy, label='stored_energy', marker='o')
plt.plot(disp, dissipated_energy, label='dissipated_energy', marker='s')
plt.plot(disp, dissipated_energy+stored_energy, label='total_energy',
        marker='x')
plt.xlabel('disp')
plt.ylabel('energy')
plt.grid(True)
plt.legend()
plt.show()

