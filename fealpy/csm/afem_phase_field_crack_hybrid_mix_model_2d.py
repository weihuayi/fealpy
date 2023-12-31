import numpy as np
import numpy as np

from ..functionspace import LagrangeFESpace
from ..fem import BilinearForm, LinearForm

from ..fem import ScalarDiffusionIntegrator
from ..fem import ScalarMassIntegrator
from ..fem import ScalarSourceIntegrator
from ..fem import ProvidesSymmetricTangentOperatorIntegrator
from ..fem import LinearElasticityOperatorIntegrator

from ..fem import DirichletBC
from ..fem import LinearRecoveryAlg
from ..mesh.adaptive_tools import mark

from scipy.sparse.linalg import spsolve

class AFEMPhaseFieldCrackHybridMixModel2d():
    """
    @brief 线性自适应有限元相场方法混合模型求解裂纹传播问题
    """
    def __init__(self, model, mesh, p=1):
        """
        @brief 

        @param[in] model 算例模型
        @param[in] mesh 连续体离散网格
        @param[in] p 有限元空间次数
        """
        self.model = model
        self.mesh = mesh
        self.p = p
        self.GD = mesh.geo_dimension()

        NC = mesh.number_of_cells()

        self.space = LagrangeFESpace(mesh, p=p)

        self.uh = self.space.function(dim=self.GD) # 位移场
        self.d = self.space.function() # 相场
        self.H = np.zeros(NC) # 最大历史应变场
        
        disp = model.is_boundary_disp()

        self.recovery = LinearRecoveryAlg()

    def newton_raphson(self, 
            disp, 
            dirichlet_phase=False, 
            refine='nvp', 
            maxit=100,
            theta=0.2):
        """
        @brief 给定位移条件，用 Newton Raphson 方法求解
        """

        mesh = self.mesh
        space = self.space
        model = self.model
        GD = self.GD
        D0 = self.linear_tangent_matrix()
        k = 0
        while k < maxit:
            print('k:', k)
            uh = self.uh
            d = self.d
            H = self.H
            
            node = mesh.entity('node')
            isDDof = model.is_disp_boundary(node)
            uh[isDDof] = disp
            du = space.function(dim=GD)

            # 求解位移
            vspace = (GD*(space, ))
            ubform = BilinearForm(GD*(space, ))

            D = self.dsigma_depsilon(d, D0)
            integrator = ProvidesSymmetricTangentOperatorIntegrator(D, q=4)
            ubform.add_domain_integrator(integrator)
            A0 = ubform.assembly()
            R0 = -A0@uh.flat[:]
            
            self.force = np.sum(-R0[isDDof.flat])
            
            ubc = DirichletBC(vspace, 0, threshold=model.is_dirchlet_boundary)

            # 这里为什么做两次边界条件处理？
            A0, R0 = ubc.apply(A0, R0) 
            A0, R0 = ubc.apply(A0, R0, dflag=isDDof)
           
            # TODO：更快的求解方法
            du.flat[:] = spsolve(A0, R0)
            uh[:] += du
            
            # 更新参数
            strain = self.strain(uh)
            phip, _ = self.strain_energy_density_decomposition(strain)
            H[:] = np.fmax(H, phip)

            # 计算相场模型
            dbform = BilinearForm(space)
            dbform.add_domain_integrator(ScalarDiffusionIntegrator(c=model.Gc*model.l0,
                q=4))
            dbform.add_domain_integrator(ScalarMassIntegrator(c=2*H+model.Gc/model.l0, q=4))
            # TODO：快速组装程序
            A1 = dbform.assembly()

            lform = LinearForm(space)
            lform.add_domain_integrator(ScalarSourceIntegrator(2*H, q=4))
            R1 = lform.assembly()
            R1 -= A1@d[:]

            if dirichlet_phase:
                dbc = DirichletBC(space, 0, threshold=model.is_boundary_phase)
                A1, R1 = dbc.apply(A1, R1)

            # TODO：快速求解程序
            d[:] += spsolve(A1, R1)
        
            self.stored_energy = self.get_stored_energy(phip, d)
            self.dissipated_energy = self.get_dissipated_energy(d)
            
            self.uh = uh
            self.d = d
            self.H = H

            # 恢复型后验误差估计子 TODO：是否也应考虑位移的奇性
            eta = self.recovery.recovery_estimate(self.d)
                
            isMarkedCell = mark(eta, theta = theta) # TODO：

            cm = mesh.cell_area() 
            isMarkedCell = np.logical_and(isMarkedCell, np.sqrt(cm) > model.l0/8)
            
            if np.any(isMarkedCell):
                if refine == 'nvp':
                    self.bisect_refine(isMarkedCell)
                elif refine == 'rg':
                    self.redgreen_refine(isMarkedCell)
            
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
        

    def bisect_refine(self, isMarkedCell):
        """
        @brief 二分法加密策略
        """
        GD = self.GD
        if GD == 2:
            data = {'uh0':self.uh[:, 0], 'uh1':self.uh[:, 1], 'd':self.d,
                    'H':self.H}
        elif GD == 3:
            data = {'uh0':self.uh[:, 0], 'uh1':self.uh[:, 1], 
                    'ud2':self.uh[:, 2], 'd':self.d, 'H':self.H}
        option = self.mesh.bisect_options(data=data, disp=False)
        self.mesh.bisect(isMarkedCell, options=option)
        print('mesh refine')      
       
        # 更新加密后的空间
        self.space = LagrangeFESpace(self.mesh, p=self.p)
        NC = self.mesh.number_of_cells()
        self.uh = self.space.function(dim=self.GD)
        self.d = self.space.function()
        self.H = np.zeros(NC, dtype=np.float64)  # 分片常数

        self.uh[:, 0] = option['data']['uh0']
        self.uh[:, 1] = option['data']['uh1']
        if GD == 3:
            self.uh[:, 2] = option['data']['uh2']
        self.d[:] = option['data']['d']
        self.H = option['data']['H']
   
    def redgreen_refine(self, isMarkedCell):
        """
        @brief 红绿加密策略
        """
        self.mesh.celldata['H'] = self.H
        mesho = copy.deepcopy(self.mesh)
        spaceo = LagrangeFESpace(mesho, p=1, doforder='vdims')
        uh0 = spaceo.function()
        uh1 = spaceo.function()
        d0 = spaceo.function()
        uh0[:] = self.uh[:, 0]
        uh1[:] = self.uh[:, 1]
        d0[:] = self.d[:]

        self.mesh.refine_triangle_rg(isMarkedCell)
        print('mesh refine')      
       
        # 更新加密后的空间
        self.space = LagrangeFESpace(self.mesh, p=1, doforder='vdims')
        NC = self.mesh.number_of_cells()
        self.uh = self.space.function(dim=self.GD)
        self.d = self.space.function()
        self.H = np.zeros(NC, dtype=np.float64)  # 分片常数
        
        self.uh[:, 0] = self.space.interpolation_fe_function(uh0)
        self.uh[:, 1] = self.space.interpolation_fe_function(uh1)
        
        self.d[:] = self.space.interpolation_fe_function(d0)
        
        self.mesh.interpolation_cell_data(mesho, datakey=['H'])
        print('interpolation cell data:', NC)      

    def strain(self, uh):
        """
        @brief 给定一个位移，计算相应的应变，这里假设是线性元
        """
        mesh = self.mesh
        mesh = self.mesh
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        gphi = mesh.grad_lambda()  # NC x 3 x 2
        GD = self.GD

        s = np.zeros((NC, GD, GD), dtype=np.float64)
        if uh.space.doforder == 'sdofs':
            uh = uh.T
        for i in range(GD):
            for j in range(i, GD):
                if i ==j:
                    s[:, i, i] = np.sum(uh[:, i][cell] * gphi[:, :, i], axis=-1)
                else:
                    val = np.sum(uh[:, i][cell] * gphi[:, :, j], axis=-1)
                    val += np.sum(uh[:, j][cell] * gphi[:, :, i], axis=-1)
                    val /= 2.0
                    s[:, i, j] = val
                    s[:, j, i] = val
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
        
        GD = self.GD
        for i in range(GD):
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

        lam = self.model.lam
        mu = self.model.mu

        # 应变正负分解
        sp, sm = self.strain_pm_eig_decomposition(s)

        ts = np.trace(s, axis1=1, axis2=2)
        tp, tm = self.macaulay_operation(ts)
        tsp = np.trace(sp**2, axis1=1, axis2=2)
        tsm = np.trace(sm**2, axis1=1, axis2=2)

        phi_p = lam * tp ** 2 / 2.0 + mu * tsp
        phi_m = lam * tm ** 2 / 2.0 + mu * tsm
        return phi_p, phi_m
    
    def dsigma_depsilon(self, phi, D0):
        """
        @brief 计算应力关于应变的导数矩阵
        @param phi 单元重心处的相场函数值, (NC, )
        @param uh 位移
        @return D 单元刚度系数矩阵
        """
        eps = 1e-10

        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        c0 = (1 - phi(bc)) ** 2 + eps
        D = np.einsum('i, jk -> ijk', c0, D0)
        return D
    
    def linear_tangent_matrix(self):
        lam = self.model.lam # 拉梅第一参数
        mu = self.model.mu # 拉梅第二参数
        mesh = self.mesh
        GD = mesh.geo_dimension()
        n = GD*(GD+1)//2
        D = np.zeros((n, n), dtype=np.float_)
        for i in range(n):
            D[i, i] += mu
        for i in range(GD):
            for j in range(i, GD):
                if i == j:
                    D[i, i] += mu+lam
                else:
                    D[i, j] += lam
                    D[j, i] += lam
        return D


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

    def get_dissipated_energy(self, d):
        model = self.model
        bc = np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
        mesh = self.mesh
        cm = mesh.entity_measure('cell')
        g = d.grad_value(bc)

        val = model.Gc/2/model.l0*(d(bc)**2+model.l0**2*np.sum(g*g, axis=1))
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

