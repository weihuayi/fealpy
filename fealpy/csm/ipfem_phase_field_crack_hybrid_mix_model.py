import numpy as np
import time
import copy

from ..functionspace import LagrangeFESpace, InteriorPenaltyBernsteinFESpace2d
from ..fem import BilinearForm, LinearForm


from ..fem import ScalarBiharmonicIntegrator
from ..fem import ScalarInteriorPenaltyIntegrator
from ..fem import ScalarDiffusionIntegrator
from ..fem import ScalarMassIntegrator
from ..fem import ScalarSourceIntegrator
from ..fem import ProvidesSymmetricTangentOperatorIntegrator
from ..fem import LinearElasticityOperatorIntegrator

from ..fem import DirichletBC
from ..fem import LinearRecoveryAlg
from ..mesh.adaptive_tools import mark
from ..mesh.halfedge_mesh import HalfEdgeMesh2d
from ..mesh import TriangleMesh

from scipy.sparse.linalg import spsolve, lgmres, cg
from scipy.sparse import csr_matrix, coo_matrix

from ..ml import timer

class IPFEMPhaseFieldCrackHybridMixModel():
    """
    @brief 内罚有限元相场方法混合模型求解裂纹传播问题
    """
    def __init__(self, model, mesh, p=1, p0=2, gamma=5):
        """
        @brief 

        @param[in] model 算例模型
        @param[in] mesh 连续体离散网格
        @param[in] p 有限元空间次数
        """
        self.model = model
        self.mesh = mesh
        self.p = p
        self.p0 = p0
        self.gamma = gamma
        self.GD = mesh.geo_dimension()

        NC = mesh.number_of_cells()

        self.space = LagrangeFESpace(mesh, p=p)
        self.dspace = LagrangeFESpace(mesh, p=p0)
        self.ipspace = InteriorPenaltyBernsteinFESpace2d(mesh, p = p0)

        self.uh = self.space.function(dim=self.GD) # 位移场
        self.d = self.dspace.function() # 相场
        self.H = np.zeros(NC) # 最大历史应变场
        self.D0 = self.linear_tangent_matrix()
        
        disp = model.is_boundary_disp()

        self.recovery = LinearRecoveryAlg()
        
        self.tmr = timer()
        next(self.tmr)

    def newton_raphson(self, 
            disp, 
            dirichlet_phase=False, 
            refine='nvp', 
            maxit=100,
            theta=0.2,
            atype='fast',
            solve='lgmres'):
        """
        @brief 给定位移条件，用 Newton Raphson 方法求解
        """
        tmr = self.tmr
        model = self.model
        GD = self.GD
        D0 = self.D0
        gamma = self.gamma
        k = 0
        while k < maxit:
            mesh = self.mesh
            space = self.space
            dspace = self.dspace
            ipspace = self.ipspace
            print('k:', k)
            uh = self.uh
            d = self.d
            H = self.H

            tmr.send('init')

            node = mesh.entity('node')
            isDDof = model.is_disp_boundary(node)
            uh[isDDof] = disp

            du = np.zeros_like(uh)
             
            # 求解位移
            vspace = (GD*(space, ))
            ubform = BilinearForm(GD*(space, ))
            tmr.send('mesh_and_space')            

            gd = self.energy_degradation_function(d)
            ubform.add_domain_integrator(LinearElasticityOperatorIntegrator(model.lam,
                model.mu, c=gd))
            tmr.send('uforms')

            if atype == 'fast':
                # 无数值积分矩阵组装
                A0 = ubform.fast_assembly()
            else:
                A0 = ubform.assembly()

            # 使用切算子计算来组装单元刚度矩阵，暂时仅能计算二维
#            D = self.dsigma_depsilon(d, D0)
#            integrator = ProvidesSymmetricTangentOperatorIntegrator(D)
#            ubform.add_domain_integrator(integrator)
#            A0 = ubform.assembly()

            R0 = -A0@uh.flat[:]
            tmr.send('uassembly') 

            self.force = np.sum(-R0[isDDof.flat])
            
            ubc = DirichletBC(vspace, 0, threshold=model.is_dirchlet_boundary)

            A0, R0 = ubc.apply(A0, R0) 
            A0, R0 = ubc.apply(A0, R0, dflag=isDDof)
            tmr.send('udirichlet')

            # 选择合适的解法器
            if solve == 'spsolve':
                du.flat[:] = spsolve(A0, R0)
            elif solve == 'lgmres':
                du.fBerrorlat[:],_ = lgmres(A0, R0, atol=1e-18)
            elif solve == 'cg':
                du.flat[:],_ = cg(A0, R0, atol=1e-18)
            elif solve == 'gpu':
                from ..solver.cupy_solver import CupySolver
                Solver = CupySolver()
                du.flat[:] = Solver.cg_solver(A0, R0, atol=1e-18)
            else:
                print("We don't have this solver yet")
            tmr.send('usolve')

            uh[:] += du
            
            # 更新应变和最大历史应变场参数
            strain = self.strain(uh)
            phip, _ = self.strain_energy_density_decomposition(strain)
            H[:] = np.fmax(H, phip)

            tmr.send('H_strain')

            # 相场模型计算
            ipbform = BilinearForm(ipspace)
            ipbform.add_domain_integrator(ScalarBiharmonicIntegrator())
            A = ipbform.assembly()
            
            P0 = ScalarInteriorPenaltyIntegrator(gamma=gamma)
            P  = P0.assembly_face_matrix(ipspace)  
            A  = model.Gc*model.l0**3/16*(A + P)
            
            tmr.send('ipMartix')

            dbform = BilinearForm(dspace)
            dbform.add_domain_integrator(ScalarDiffusionIntegrator(c=model.Gc*model.l0))
            dbform.add_domain_integrator(ScalarMassIntegrator(c=2*H+model.Gc/model.l0/2.0))
            A1 = dbform.assembly()
            tmr.send('dlMatrix')


            '''
            start2 = time.time()
            if atype == 'fast':
                # 无数值积分矩阵组装
                A1 = dbform.fast_assembly()
            else:
                A1 = dbform.assembly()
            '''
            A1 += A

            # 线性积分子
            lform = LinearForm(dspace)
            lform.add_domain_integrator(ScalarSourceIntegrator(2*H, q=4))
            R1 = lform.assembly()
            R1 -= A1@d[:]
            
            tmr.send('dlvector')

            if dirichlet_phase:
                dbc = DirichletBC(dspace, 0, threshold=model.is_boundary_phase)
                A1, R1 = dbc.apply(A1, R1)
            
            tmr.send('d_dirichlet')

            # 选择合适的解法器
            if solve == 'spsolve':
                dd = spsolve(A1, R1)
            elif solve == 'lgmres':
                dd,_ = lgmres(A1, R1, atol=1e-20)
            elif solve == 'cg':
                dd,_ = cg(A1, R1, atol=1e-20)
            elif solve == 'gpu':
                dd = Solver.gmres_solver(A1, R1, atol=1e-20)
            else:
                print("We don't have this solver yet")
            d[:] += dd
           
            tmr.send('dsolve')

            self.stored_energy = self.get_stored_energy(phip, d)
            self.dissipated_energy = self.get_dissipated_energy(d)
            
            tmr.send('energy')

            self.uh = uh
            self.d = d
            self.H = H

            # 恢复型后验误差估计子 TODO：是否也应考虑位移的奇性
            eta = self.recovery.recovery_estimate(self.d)
                
            isMarkedCell = mark(eta, theta = theta) # TODO：
            
            tmr.send('Markcell')

            cm = mesh.entity_measure('cell')
            if GD == 3:
                hmin = model.l0**3/200
            else:
                hmin = (model.l0/8)**2
            isMarkedCell = np.logical_and(isMarkedCell, cm > hmin)
            if np.any(isMarkedCell):
                if GD == 2:
                    if refine == 'nvp':
                        self.bisect_refine_2d(isMarkedCell)
                    elif refine == 'rg':
                        self.redgreen_refine_2d(isMarkedCell)
                elif GD == 3:
                        self.bisect_refine_3d(isMarkedCell)
                else:
                    print("GD is not 2 or 3, it is incorrect")
            
            tmr.send('refine')

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
            tmr.send('stop')
    
    def bisect_refine_2d(self, isMarkedCell):
        """
        @brief 二分法加密策略
        """
        dcell2dof = self.dspace.cell_to_dof()
        data = {'uh0':self.uh[:, 0], 'uh1':self.uh[:, 1], 'd':self.d[dcell2dof],
                'H':self.H}
        option = self.mesh.bisect_options(data=data, disp=False)
        self.mesh.bisect(isMarkedCell, options=option)
        print('mesh refine')      
       
        # 更新加密后的空间
        self.space = LagrangeFESpace(self.mesh, p=self.p)
        self.dspace = LagrangeFESpace(self.mesh, p=self.p0)
        self.ipspace = InteriorPenaltyBernsteinFESpace2d(self.mesh, p = self.p0)

        NC = self.mesh.number_of_cells()
        self.uh = self.space.function(dim=self.GD)
        self.d = self.dspace.function()
        self.H = np.zeros(NC, dtype=np.float64)  # 分片常数
        
        dcell2dof = self.dspace.cell_to_dof()
        self.uh[:, 0] = option['data']['uh0']
        self.uh[:, 1] = option['data']['uh1']
        self.d[dcell2dof.reshape(-1)] = option['data']['d'].reshape(-1)
        self.H = option['data']['H']
   
    def redgreen_refine_2d(self, isMarkedCell):
        """
        @brief 红绿加密策略
        """
        mesh0 = HalfEdgeMesh2d.from_mesh(self.mesh, NV=3) 

#        mesh0.celldata['H'] = self.H
        mesho = copy.deepcopy(mesh0)
        spaceo = LagrangeFESpace(mesho, p=1, doforder='vdims')
        dspaceo = LagrangeFESpace(mesho, p=self.p0)

        uh0 = spaceo.function()
        uh1 = spaceo.function()
        d0 = dspaceo.function()
        uh0[:] = self.uh[:, 0]
        uh1[:] = self.uh[:, 1]
        d0[:] = self.d[:]

        mesh0.refine_triangle_rg(isMarkedCell)
        print('mesh refine')

        node = mesh0.entity(etype='node')
        cell = mesh0.entity(etype='cell')
        mesh = TriangleMesh(node[:], cell[:])
        self.mesh = mesh
       
        # 更新加密后的空间
        self.space = LagrangeFESpace(self.mesh, p=1, doforder='vdims')
        self.dspace = LagrangeFESpace(self.mesh, p=self.p0)
        self.ipspace = InteriorPenaltyBernsteinFESpace2d(self.mesh, p = self.p0)

        space0 = LagrangeFESpace(mesh0, p=1, doforder='vdims')
        dspace0 = LagrangeFESpace(mesh0, p=self.p0)

        NC = self.mesh.number_of_cells()
        self.uh = self.space.function(dim=self.GD)
        self.d = self.dspace.function()
        self.H = np.zeros(NC, dtype=np.float64)  # 分片常数
        
        self.uh[:, 0] = space0.interpolation_fe_function(uh0)
        self.uh[:, 1] = space0.interpolation_fe_function(uh1)
        
        self.d[:] = dspace0.interpolation_fe_function(d0)
        
        self.H[:] = mesh0.interpolation_cell_data(mesho, datakey=['H'])
        print('interpolation cell data:', NC)      
        
    def energy_degradation_function(self, d):
        eps = 1e-10
        gd = np.zeros_like(d)
        qf = self.mesh.integrator(self.p+1, 'cell')
        bc, ws = qf.get_quadrature_points_and_weights()
        gd = (1 - d(bc)) ** 2 + eps
        return gd

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
        lam = self.model.lam # 拉梅第一参数
        mu = self.model.mu # 拉梅第二参数
        
        qf = self.mesh.integrator(self.p+1, 'cell')
        bc, ws = qf.get_quadrature_points_and_weights()

        c0 = (1 - phi(bc)) ** 2 + eps
        D = np.einsum('i, jk -> ijk', c0, D0)
#        c0 = (1 - phi(bc)) ** 2 + eps
#        D0 = np.array([[2*mu+lam, lam, 0], [lam, 2*mu+lam, 0], [0, 0, mu]],
#                dtype=np.float_)
#        D = np.einsum('i, jk -> ijk', c0, D0)
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
        mesh = self.mesh
        qf = mesh.integrator(self.p, 'cell')
        bc, ws = qf.get_quadrature_points_and_weights()
        cm = mesh.entity_measure('cell')
        g = d.grad_value(bc)

        val = model.Gc/2/model.l0*(d(bc)**2+model.l0**2*np.sum(g*g, axis=-1))
        dissipated = np.einsum('q, qc, c->', ws, val, cm)
        return dissipated

    
    def get_stored_energy(self, psi_s, d):
        eps = 1e-10
        mesh = self.mesh

        qf = mesh.integrator(self.p, 'cell')
        bc, ws = qf.get_quadrature_points_and_weights()
        c0 = (1 - d(bc)) ** 2 + eps
        cm = mesh.entity_measure('cell')
        val = c0*psi_s
        stored = np.einsum('q, qc, c->', ws, val, cm)
        return stored

    def stress(self):
        '''
        @brief 计算每个单元的应力张量
        '''
        uh = self.uh
        gd = self.energy_degradation_function(self.d)
        lam = self.model.lam
        mu = self.model.mu
        strains = self.strain(uh)
        trace_e = np.trace(strains, axis1=1, axis2=2)

        # 构造单位张量数组
        eye = np.eye(strains.shape[1])

        # 计算每个单元的应力张量
        stresses = lam * trace_e[:, None, None] * eye + 2 * mu * strains
        stresses = gd[:, None, None]*stresses
        return stresses


