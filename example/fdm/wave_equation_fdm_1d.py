import ipdb
from matplotlib import pyplot as plt

from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')
from fealpy.pde.wave_1d import StringOscillationPDEData
from fealpy.mesh import UniformMesh
from fealpy.fdm.wave_operator import WaveOperator
from fealpy.fdm.dirichlet_bc import DirichletBC
from fealpy.solver import spsolve
from functools import partial

def wave_1d_run(nx: int, nt: int, time_step: str='explicit', mode: str='animation', theta: float=0.25, 
                fixed_time: float=0.5, maxit: int=5, show_plot: bool=True):
    """
    求解一维波动方程
    
    参数:
    nx: int, 空间剖分数量
    nt: int, 时间剖分数量
    time_step: str, 时间步进格式，可选 'explicit' 或 'implicit'
    mode: str, 计算模式，可选 'animation' 或 'convergence'
    theta: float, 隐格式的参数, 仅当time_step='implicit'时使用
    fixed_time: float, 收敛性分析的固定时刻, 仅当mode='convergence'时使用
    maxit: int, 收敛性分析的最大迭代次数, 仅当mode='convergence'时使用
    show_plot: bool, 是否显示图像
    
    返回:
    根据mode返回不同结果:
    - 'animation': 返回(fig, axes)
    - 'convergence': 返回(em, convergence_rates)
    """
    # 初始化PDE问题
    pde = StringOscillationPDEData()
    domain = pde.domain()
    duration = pde.duration()
    
    # 空间离散
    hx = (domain[1] - domain[0])/nx
    extent = [0, nx]
    mesh = UniformMesh(domain, extent)
    
    # 时间离散
    tau = (duration[1] - duration[0])/nt
    
    # 准备初值
    uh0 = mesh.interpolate(pde.init_solution, 'node').flatten()
    vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node').flatten()
    uh1 = mesh.function('node').flatten()
    
    # 定义步进函数（内部函数，捕获外部环境）
    def advance_explicit(n, *frags):
        """时间步进格式为显格式"""
        nonlocal uh0, uh1, vh0
        
        t = duration[0] + n*tau
        if n == 0:
            return uh0, t
        elif n == 1:
            rx = tau/hx
            uh1[1:-1] = rx**2*(uh0[0:-2] + uh0[2:])/2.0 + (1-rx**2)*uh0[1:-1] + tau*vh0[1:-1]
            gD = lambda p: pde.dirichlet(p, t)
            mesh.update_dirichlet_bc(gD, uh1)
            return uh1, t
        else:
            wave_operator = WaveOperator(mesh)
            A = wave_operator.assembly(tau=tau)
            source = lambda p: pde.source(p, t + tau)
            f = mesh.interpolate(source)
            f *= tau**2
            uh2 = A@uh1 - uh0 + f  # 根据差分方程进行更新
            gD = lambda p: pde.dirichlet(p, t + tau)
            mesh.update_dirichlet_bc(gD, uh2)
            uh0[:] = uh1
            uh1[:] = uh2
            
            return uh1, t

    def advance_implicit(n, theta=theta, *frags):
        """时间步进格式为隐格式"""
        nonlocal uh0, uh1, vh0
        
        t = duration[0] + n*tau
        if n == 0:
            return uh0, t
        elif n == 1:
            rx = tau/hx 
            uh1[1:-1] = rx**2*(uh0[0:-2] + uh0[2:])/2.0 + (1-rx**2)*uh0[1:-1] + tau*vh0[1:-1]
            gD = lambda p: pde.dirichlet(p, t)
            mesh.update_dirichlet_bc(gD, uh1)
            return uh1, t
        else:
            wave_operator = WaveOperator(mesh)
            A0, (A1, A2) = wave_operator.implicit_assembly(tau=tau, theta=theta)
            source = lambda p: pde.source(p, t + tau)
            f = mesh.interpolate(source)
            f *= tau**2
            f += A1@uh1 + A2@uh0

            uh0[:] = uh1[:]
            gD = lambda p: pde.dirichlet(p, t + tau)

            dirichlet_bc = DirichletBC(mesh, gD)
            A0, f = dirichlet_bc.apply(A0, f)
            uh1[:] = spsolve(A0, f, solver='scipy')
                
            return uh1, t

    # 根据计算模式执行不同的计算
    if mode == 'animation':
        box = [0, 1, -1.5, 1.5]
        fig, axes = plt.subplots()

        if time_step == 'explicit':
            mesh.show_animation(fig, axes, box, advance_explicit, fname=f'{time_step}.mp4', frames=nt+1)
        elif time_step == 'implicit':
            advance_implicit_theta = partial(advance_implicit, theta=theta)
            mesh.show_animation(fig, axes, box, advance_implicit_theta, fname=f'{time_step}.mp4', frames=nt+1)
        else:
            raise ValueError(f"未知的时间步进格式: {time_step}")
            
        if show_plot:
            plt.show()
            
        return fig, axes
        
    elif mode == 'convergence':
        # 收敛性分析
        tau_init = tau  # 初始时间步长
        em = bm.zeros((3, maxit), dtype=bm.float64)
        h_sizes = bm.zeros(maxit, dtype=bm.float64)
        
        for i in range(maxit):
            # 记录当前网格尺寸
            hx = (domain[1] - domain[0])/mesh.nx
            h_sizes[i] = hx

            # 调整时间步长与网格尺寸
            if i > 0:
                tau = tau_init * (h_sizes[i]/h_sizes[0])

            fixed_time_step = int((fixed_time - duration[0]) / tau)
            
            # 重新初始化解
            uh0 = mesh.interpolate(pde.init_solution, 'node').flatten()
            vh0 = mesh.interpolate(pde.init_solution_diff_t, 'node').flatten()
            uh1 = mesh.function('node').flatten()

            # 计算到指定时刻
            for n in range(fixed_time_step + 1):
                if time_step == 'explicit':
                    uh, t = advance_explicit(n)
                elif time_step == 'implicit':
                    uh, t = advance_implicit(n)
                else:
                    raise ValueError(f"未知的时间步进格式: {time_step}")

            # 计算误差
            solution = lambda p: pde.solution(p, fixed_time)
            em[0, i], em[1, i], em[2, i] = mesh.error(solution, uh1)
            
            # 网格加密（除了最后一次迭代）
            if i < maxit - 1:
                mesh.uniform_refine()

        # 计算收敛阶
        convergence_rates = bm.log2(em[:, 0:-1]/em[:, 1:])
        
        print("网格尺寸:\n", h_sizes)
        print("误差:\n", em)
        print("误差比值:\n", em[:, 0:-1]/em[:, 1:])
        print("收敛阶:\n", convergence_rates)
        
        # 可视化收敛性结果
        if show_plot:
            fig, ax = plt.subplots()
            ax.loglog(h_sizes, em[0, :], 'o-', label='$L_{\infty}$ Error')
            ax.loglog(h_sizes, em[1, :], 's-', label='$L_2$ Error')
            ax.loglog(h_sizes, em[2, :], '^-', label='$H_1$ Error')
            
            # Add reference line for second-order convergence
            hx_ref = bm.linspace(h_sizes[-1]/2, h_sizes[0]*2, 100)
            ax.loglog(hx_ref, hx_ref**2, '--', label='$O(h^2)$')
            
            ax.set_xlabel('Mesh Size h')
            ax.set_ylabel('Error')
            ax.legend()
            ax.set_title(f'Convergence Analysis at t={fixed_time} ({time_step} scheme)')
            plt.show()
        
        return em, convergence_rates
    
    else:
        raise ValueError(f"未知的计算模式: {mode}")

if __name__ == "__main__":
    # 示例1：生成动画
    # fig, ax = wave_1d_run(nx=400, nt=500, time_step='explicit', mode='animation')
    
    # 示例2：收敛性分析
    em, rates = wave_1d_run(nx=50, nt=100, time_step='explicit', mode='convergence', 
                            theta=0.25, fixed_time=0.5, maxit=5)
