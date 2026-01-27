from fealpy.backend import bm
from fealpy.fem import (ScalarDiffusionIntegrator,ScalarConvectionIntegrator,
                        ScalarSourceIntegrator,ScalarMassIntegrator)
from fealpy.fem import LinearForm, BilinearForm , DirichletBC
from fealpy.functionspace import LagrangeFESpace
from fealpy.mmesh import MMesher, Config
from fealpy.decorator import barycentric
from fealpy.solver import spsolve
from fealpy.mmesh.pde.scalar_burgers_data import ScalarBurgersData
import matplotlib.pyplot as plt

class Burgers_MMsolver:
    def __init__(self, pde: ScalarBurgersData ,p = 1 , 
                 nt = 500 , method = 'default', sub_steps=4):
        """
        标量Burgers方程移动网格求解器, 时间积分采用SDIRK2方法
        u_t + u u_x + u u_y - 1/Re (u_xx + u_yy) = f
        
        Parameters
            pde : ScalarBurgersData
               标量Burgers方程数据对象
            p : int
                有限元空间的多项式次数
            nt : int
                时间步数
            method : str
                移动网格方法
        """
        self.pde = pde
        self.nt = nt
        self.method = method
        
        self.dt = pde.T[1]/nt
        self.p = p
        self.q = p + 2
        self.mesh = pde.mesh
        self.Re = pde.Re
        
        gamma = 1- bm.sqrt(2)/2 
        self.tau1 = gamma
        self.tau2 = 1
        self.a11 = gamma
        self.a21 = 1- gamma
        self.a22 = gamma
        self.b1 = 1 - gamma
        self.b2 = gamma
        self.sub_steps = sub_steps
    
    def linear_system(self):
        """
        线性系统的组装和函数空间的定义
        """
        self.space = LagrangeFESpace(self.mesh, p=self.p)
        self.bform0 = BilinearForm(self.space)
        self.bform1 = BilinearForm(self.space)
        self.lform = LinearForm(self.space)
        
        # 积分子定义
        self.SSI = ScalarSourceIntegrator(q=self.q)
        self.SMI = ScalarMassIntegrator(q=self.q)
        self.SDI = ScalarDiffusionIntegrator(q=self.q)
        self.SCI = ScalarConvectionIntegrator(q=self.q)
        
        self.bform0.add_integrator(self.SDI,self.SCI)
        self.bform1.add_integrator(self.SMI)
        self.lform.add_integrator(self.SSI)
        
        self.uh = self.space.function()
        self.u1 = self.space.function()
        self.u2 = self.space.function()
        
        self.bc = DirichletBC(self.space)
    
    def moving_mesher(self):
        """
        移动网格方法的初始化
        """
        method = self.method
        
            
        mesh = self.mesh
        space = self.space
        config = Config()
        if method == 'default':
            config.active_method = 'GFMMPDE'
        else:
            config.active_method = method
        config.is_pre = False
        # config.pde = pde
        config.mol_times = 5
        config.pre_steps = 3
        config.alpha = 0.5
        config.tau = 0.005
        config.t_max = 0.5
        uh0 = space.interpolate(self.pde.init_solution)
        self.mm = MMesher(mesh,uh=uh0, space=space,beta=0.5, config=config)
        self.mm.initialize()
        self.mm.set_interpolation_method('linear')
        self.mm.set_monitor('linear_int_error')
        self.mm.set_mol_method('huangs_method')
        self.mspace = self.mm.instance.mspace
    
    def update(self,uh , t ,mv , sub_steps=None):
        if sub_steps is None:
            sub_steps = self.sub_steps
        delta = self.dt / sub_steps
        mesh = self.mesh
        a = 1 /(2 * self.Re)
        SDI = self.SDI
        SCI = self.SCI
        SSI = self.SSI
        SMI = self.SMI
        bc = self.bc
        space = self.space
        node0 = mesh.node - mv * self.dt
        v0 = self.mspace.function(mv[:,0])
        v1 = self.mspace.function(mv[:,1])
        
        for j in range(sub_steps):
            t_hat = t + j * delta
            mesh.node = node0 + self.tau1 * mv * delta
            M = self.bform1.assembly()
            
            SDI.coef = a * delta * self.a11
            SMI.coef = 1.0
            
            @barycentric
            def coef1(bcs, index):
                v0_val = v0(bcs, index)
                v1_val = v1(bcs, index)
                v_value = bm.concat([v0_val[...,None], v1_val[...,None]], axis=-1)
                return -delta * self.a11 * v_value
            SCI.coef = coef1
            @barycentric
            def source1(bcs , index):
                guh = uh.grad_value(bcs , index)
                result = -guh[...,0]  - guh[...,1]
                result *= delta * self.a11 * uh(bcs , index)
                result += uh(bcs , index)
                return result
            SSI.source = source1
            
            A = self.bform0.assembly()
            A += M
            
            b = self.lform.assembly()
            bc.gd = lambda p: self.pde.dirichlet(p , t_hat + self.tau1 * delta)
            A,b = bc.apply(A , b)
            self.u1[:] = spsolve(A , b , 'scipy')
            
            mesh.node = node0 + delta * mv
            k1 = (self.u1 - uh) / (self.tau1 * delta)
            k1 = space.function(k1)

            SDI.coef = a * delta * self.a22
            SMI.coef = 1.0
            
            @barycentric
            def coef2(bcs, index):
                v0_val = v0(bcs, index)
                v1_val = v1(bcs, index)
                v_value = bm.concat([v0_val[...,None], v1_val[...,None]], axis=-1)
                return - delta * self.a22 * v_value
            SCI.coef = coef2
            @barycentric
            def source2(bcs , index):
                guh = uh.grad_value(bcs , index)
                result = -guh[...,0] - guh[...,1]
                result *= delta * self.a22 * uh(bcs , index)
                result += uh(bcs , index) + delta * self.a21 * k1(bcs , index)
                return result
            SSI.source = source2
            A = self.bform0.assembly()
            M = self.bform1.assembly()
            A += M
            
            b = self.lform.assembly()
            bc.gd = lambda p: self.pde.dirichlet(p , t_hat + self.tau2 * delta)
            A,b = bc.apply(A , b)
            self.u2[:] = spsolve(A , b , 'scipy')
            
            k2 = (self.u2 - uh - delta * self.a21 * k1) / (self.a22 * delta)
            self.uh[:] = uh + delta * (self.b1 * k1 + self.b2 * k2)
            
            node0 = mesh.node.copy()
            uh[:] = self.uh[:]
        
        return uh
    
    def error(self,uh , t):
        mesh = self.mesh
        pde = self.pde
        L2_error = mesh.error(uh,lambda p : pde.solution(p,t+self.dt),power = 2)
        uh_grad = uh.grad_value
        
        H1_error = mesh.error(uh_grad,lambda p : pde.gradient(p,t+self.dt),power = 2)
        
        return L2_error , H1_error
    
    def _plot_errors(self, L2_list, H1_list, times):
        """绘制 L2 和 H1 误差随时间的变化曲线"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=100)
        
        # L2 误差曲线
        ax1.semilogy(times, L2_list, 'b-o', linewidth=2, markersize=4, label='L2 Error')
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('L2 Error', fontsize=12)
        ax1.set_title('L2 Error vs Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=11)
        
        # H1 误差曲线
        ax2.semilogy(times, H1_list, 'r-s', linewidth=2, markersize=4, label='H1 Error')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('H1 Error', fontsize=12)
        ax2.set_title('H1 Error vs Time', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'burgers_errors_{self.method}_nt{self.nt}.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def solve(self, vtu_path=None):
        self.linear_system()
        self.moving_mesher()
        pde = self.pde
        nt = self.nt
        dt = self.dt
        space = self.space
        mesh = self.mesh
        mm = self.mm
        mm.run()

        self.uh = space.interpolate(pde.init_solution)
        times = bm.linspace(0, pde.T[1], nt + 1)
        sub_steps = self.sub_steps * 3
        L2_list = []
        H1_list = []
        for i in range(nt):
            x0 = mesh.node.copy()
            if i>0:
                mm.run()
                sub_steps = None
                
            mv = (mesh.node - x0)/dt
            t = times[i]
            self.uh[:] = self.update(self.uh , t , mv , sub_steps=sub_steps)
            L2_error , H1_error = self.error(self.uh , t)
            L2_list.append(L2_error)
            H1_list.append(H1_error)
            mm.update_solution(self.uh)
            if vtu_path is not None:
                mesh.nodedata['u'] = self.uh[:]
                mesh.to_vtk(f'{vtu_path}_step{i+1:04d}.vtu')
            print(f'Step {i+1}/{nt}, Time={t+dt:.4f}, L2 Error={L2_error:.6e}, H1 Error={H1_error:.6e}')
        
        self._plot_errors(L2_list, H1_list, times[:-1])
        return self.uh, L2_list, H1_list    

Re = 100
u = f'1/(1+ exp((x+y-t)/({1/Re})))'
var = ['x', 'y', 't']
D = [0, 1, 0, 1]
T = [0,2]
support_method = ['default', 'GFMMPDE','Harmap', 
                  'EAGAdaptiveHuang','EAGAdaptiveXHuang',
                  'MetricTensorAdaptive', 'MetricTensorAdaptiveX']

pde = ScalarBurgersData(u, var, D, T, Re=Re)
pde.set_mesh(nx=45, ny=45, meshtype='tri')
BGMMSolver = Burgers_MMsolver(pde , p=1 , nt = 500 
                              , method='MetricTensorAdaptive',sub_steps=6)
uh, L2_list, H1_list = BGMMSolver.solve(vtu_path='burgers_solution')