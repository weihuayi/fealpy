from fealpy.backend import bm
from fealpy.cfd.example.rising_bubble.gaugeuzawa_twophase_flow_solver import GaugeUzawaTwoPhaseFlowSolver
from fealpy.cfd.example.rising_bubble.twophaseflow_withsolution_pde import TwoPhaseFlowPDE

def main():
    # 创建PDE问题
    n = 64
    pde = TwoPhaseFlowPDE()
    pde.set_mesh(nx = n, ny = n, mesh_type="uniform_tri")
    mesh = pde.mesh
    # 验证解析解的正确性
    pde.verify_phase_solution()
    pde.verify_incompressibility_solution() 
    pde.verify_momentum_solution()

    # 创建求解器
    T = 0.2      # 总时间
    dt = 0.02  # 时间步长
    solver = GaugeUzawaTwoPhaseFlowSolver(
        pde=pde, 
        mesh=mesh, 
        up=2,      # 速度空间次数
        pp=1,      # 压力空间次数  
        phip=1,    # 相场空间次数
        dt=dt,     # 时间步长
        q=4,       # 积分精度
        method=None,
    )
    
    print(f"init done.")
    print(f"velocity dof: {solver.uspace.number_of_global_dofs()}")
    print(f"pressure dof: {solver.pspace.number_of_global_dofs()}")
    print(f"phase dof: {solver.phispace.number_of_global_dofs()}")

    phi0, uh0, ph0 = solver.phi0 , solver.uh0 , solver.ph
    # 设置初始条件
    t0 = 0.0
    # 计算初始误差
    initial_errors = solver.current_error(phi0, ph0, uh0, t0)
    print(f"initial errors:")
    print(f"velocity error: {initial_errors[2]:.6e}")
    print(f"pressure error: {initial_errors[1]:.6e}")
    print(f"phase error: {initial_errors[0]:.6e}")
    
    # 运行时间演化求解
    output_freq = 1 # 输出频率
    step = 0
    print(f"Starting time evolution solution, total time: {T}, time step: {dt}")
    t = 0.0
    while t < T:
        step += 1
        t += dt

        # Execute one time step
        phi,p , u = solver.time_step(t=t, phi0=phi0, uh0=uh0)
        phi0[:] = phi
        uh0[:] = u
        errors = solver.current_error(phi, p, u, t)
        if step % output_freq == 0:
            print(f"Time step: {step}, time: {t:.7f}")
            print(f"velocity_error: {errors[2]:.6e}")
            print(f"pressure_error: {errors[1]:.6e}")
            print(f"phase_error: {errors[0]:.6e}")
                
        print(f"Solution completed, total time steps: {step}")
    
    # 计算最终误差
    final_errors = solver.current_error(phi, p, u, T)
    print(f"final_errors:")
    print(f"velocity_error: {final_errors[2]:.6e}")
    print(f"pressure_error: {final_errors[1]:.6e}")
    print(f"phase_error: {final_errors[0]:.6e}")

if __name__ == '__main__':
    main()