from ..nodetype import CNodeType, PortConf, DataType

__all__ = ['MMGUACNSFEMSolver']

class MMGUACNSFEMSolver(CNodeType):
    r"""Finite Element Discretization for the MovingMesh Gauge Uzawa ACNS Equation.

    This node constructs the finite element bilinear and linear forms
    required for the discretization of the MovingMesh Gauge Uzawa formulation of 
    the ACNS equations using the provided function spaces.
    
    Inputs:
        dt (float): Time step size.
        nt (int): Number of time steps.
        uspace (SpaceType): Finite element space for the velocity field.
        pspace (SpaceType): Finite element space for the pressure field.
        phispace (SpaceType): Finite element space for the phase field φ.
        update_ac (function): Function that assembles the Allen-Cahn system.
        update_us (function): Function that assembles the auxiliary velocity system.
        update_ps (function): Function that updates the pressure field.
        update_velocity (function): Function that updates the velocity field.
        update_gauge (function): Function that updates the gauge variable.
        update_pressure (function): Function that updates the pressure field.
        q (int): Quadrature degree used for numerical integration.
    Outputs:
        u (Function): Velocity vector field at the final time.
        ux (Function): x-component of velocity field.
        uy (Function): y-component of velocity field.
        p (Function): Pressure field.
        phi (Function): Phase-field function.
    """
    TITLE: str = "移动网格有限元求解 ACNS 方程"
    PATH: str = "simulation.solvers"
    DESC: str = """该节点实现了两相不可压流体的 Allen-Cahn-Navier-Stokes (ACNS) 方程组有限元求解
                器。ACNS 模型结合了相场法与流体力学方程，用以描述两种不可混溶流体的界面演化与流动耦合过程。
                通过时间步推进(dt, nt)，程序在每个时间步内依次执行：
                1. Allen-Cahn 方程：更新相场函数 φ 与化学势 μ；
                2. Navier-Stokes 方程：根据当前相场计算密度场 ρ(φ)，更新速度场 u 与压力场 p。

                输入参数：
                - dt (float)：时间步长；
                - nt (int)：总时间步数；
                - uspace (SpaceType)：速度场有限元空间；
                - pspace (SpaceType)：压力场有限元空间；
                - phispace (SpaceType)：相场 φ 的有限元空间；
                - update_ac (function)：组装并更新 Allen–Cahn 子问题；
                - update_us (function)：组装并更新辅助速度系统；
                - update_ps (function)：压力更新流程；
                - update_velocity (function)：速度校正/投影更新；
                - update_gauge (function)：规范变量（gauge）更新；
                - update_pressure (function)：压力更新函数；
                - q (int)：数值积分的求积阶数。

                输出结果：
                - u ：最终时刻的速度场；
                - ux、uy ：速度在 x、y 方向的分量；
                - p ：压力场；
                - phi ：相场函数。

                使用示例：
                可将“相场更新模块 (AC)”与“流体更新模块 (gu-NS)”分别连接到 `update_ac` 与 
                (`update_us`, `update_ps`, `update_velocity`, `update_gauge`, `update_p`)，
                设置好初始界面与网格信息后，即可在多时间步迭代中自动完成流体界面的演化与速度场的时序求解。
                """
    INPUT_SLOTS = [
        PortConf("domain", DataType.NONE, title="计算域"),
        PortConf("dt", DataType.FLOAT, 0, title="时间步长", default=0.001),
        PortConf("nt", DataType.INT, 0, title="时间步数", default=2000),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("phispace", DataType.SPACE, title="相场函数空间"),
        PortConf("update_ac", DataType.FUNCTION, title="相场更新函数"),
        PortConf("update_us", DataType.FUNCTION, title="辅助速度更新函数"),
        PortConf("update_ps", DataType.FUNCTION, title="伪压力更新函数"),
        PortConf("update_velocity", DataType.FUNCTION, title="速度更新函数"),
        PortConf("update_gauge", DataType.FUNCTION, title="规范变量更新函数"),
        PortConf("update_pressure", DataType.FUNCTION, title="压力更新函数"),
        PortConf("init_phase", DataType.FUNCTION, title="初始相场函数"),
        PortConf("init_velocity", DataType.FUNCTION, title="初始速度函数"),
        PortConf("init_pressure", DataType.FUNCTION, title="初始压力函数"),
        PortConf("velocity_dirichlet_bc", DataType.FUNCTION, title="速度边界条件"),
        PortConf("phase_force", DataType.FUNCTION, title="相场源项"),
        PortConf("velocity_force", DataType.FUNCTION, title="速度源项"),
        PortConf("output_dir", DataType.STRING, title="输出目录")
    ]
    OUTPUT_SLOTS = [
        PortConf("u", DataType.FUNCTION, title="速度场"),
        PortConf("ux", DataType.FUNCTION, title="速度场 x 分量"),
        PortConf("uy", DataType.FUNCTION, title="速度场 y 分量"),
        PortConf("p", DataType.FUNCTION, title="压力场"),
        PortConf("phi", DataType.FUNCTION, title="相场函数")
    ]
    
    @staticmethod
    def run(domain, dt, nt,
            uspace, pspace, phispace,
            update_ac, update_us, update_ps,
            update_velocity, update_gauge, update_pressure,
            init_phase,init_velocity,init_pressure,
            velocity_dirichlet_bc,
            phase_force,velocity_force,
            output_dir
            ):
        from fealpy.backend import bm
        from fealpy.solver import spsolve
        from fealpy.functionspace import TensorFunctionSpace
        from pathlib import Path
        mesh = uspace.mesh
        export_dir = Path(output_dir).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        
        def set_move_mesher(mesh ,domain, phi_n , phispace,
                            mmesher:str = 'GFMMPDE',
                            beta :float = 1,
                            tau :float = 0.1,
                            tmax :float = 0.5,
                            alpha :float = 0.75,
                            moltimes :int = 4,
                            monitor: str = 'arc_length',
                            mol_meth :str = 'projector',
                            config : dict = None):
            from fealpy.mmesh.mmesher import MMesher
            mesh.meshdata['vertices'] =bm.array([[domain[0], domain[2]],
                             [domain[1], domain[2]],
                             [domain[1], domain[3]],
                             [domain[0], domain[3]]], dtype=bm.float64)
            mm = MMesher(mesh, 
                        uh = phi_n ,
                        space= phispace,
                        beta=beta,
                        ) 
            mm.config.active_method = mmesher
            mm.config.tau = tau
            mm.config.t_max = tmax
            mm.config.alpha = alpha
            mm.config.mol_times = moltimes
            mm.config.monitor = monitor
            mm.config.mol_meth = mol_meth
            mm.config.is_pre = False
            if config is not None:
                for key, value in config.items():
                    # check if the key exists in the config
                    getattr(mm.config, key)
                    # if it exists, set the value
                    setattr(mm.config, key, value)

            mm.initialize()
            mm.set_interpolation_method('linear')
            node_n = mesh.node.copy()
            smspace =  mm.instance.mspace
            mspace = TensorFunctionSpace(smspace, (mesh.GD,-1))
            mesh_velocity = mspace.function()
            return mm, mesh_velocity, node_n
        
        def save_vtu(step: int ,export_dir):
            mesh.nodedata['interface'] = phi
            mesh.nodedata['velocity'] = u.reshape(mesh.GD,-1).T
            mesh.nodedata['pressure'] = p
            fname = export_dir / f"two_phase_flow_{str(step).zfill(10)}.vtu"
            mesh.to_vtk(fname=fname)
                
        def compute_bubble_centroid():
            # 获取网格节点和相场值
            nodes = mesh.node  # 网格节点坐标
            cell = mesh.cell
            cell_to_dof = phispace.cell_to_dof()
            phi_cell_values = bm.mean(phi[cell_to_dof],axis=-1)  # 相场值

            # 筛选气泡区域（phi > 0）
            bubble_mask = phi_cell_values > 0
            bubble_bc_nodes = bm.mean(nodes[cell[bubble_mask]],axis=1)
            bubble_phi = phi_cell_values[bubble_mask]

            # 计算质心位置
            centroid = bm.sum(bubble_bc_nodes.T * bubble_phi, axis=1) / bm.sum(bubble_phi)
            return centroid
        
        # Initialize functions
        u = uspace.function()  # Current velocity
        u_n = uspace.function()  # Previous time velocity
        p = pspace.function()  # Current pressure
        phi = phispace.function()  # Current phase-field
        phi_n = phispace.function()  # Previous time phase-field
        
        # Intermediate variables
        us = uspace.function()  # Intermediate velocity
        s = pspace.function()  # guage variable
        s_n = pspace.function()  # guage variable
        ps = pspace.function()  # Intermediate pressure
        
        ugdof = uspace.number_of_global_dofs()
        # Set initial velocity
        u_n[:] = uspace.interpolate(lambda p: init_velocity(p))
        u[:] = u_n[:]
        # Set initial pressure
        p[:] = pspace.interpolate(lambda p: init_pressure(p))
        # Set initial phase-field
        phi_n[:] = phispace.interpolate(lambda p: init_phase(p))
        phi[:] = phi_n[:]
        
        mm, mesh_velocity, node_n = set_move_mesher(mesh,domain, phi_n, phispace)
        t = 0.0
        for i in range(nt):
            t += dt
            # Move mesh according to phase field
            if t - dt == 0.0:
                # First time step, need to initialize the mesh movement
                mm.run()
                node_n = mesh.node.copy()
                phi_n = phispace.interpolate(lambda p: init_phase(p))
                phi[:] = phi_n[:]
                save_vtu(step = 0,export_dir=export_dir)
            else:
                mm.run()
                
            mesh_velocity[:] = ((mesh.node - node_n)/dt).T.flatten()
            # Define time-dependent forces and boundary conditions
            current_v_force = lambda p: velocity_force(p, t)
            current_phi_force = lambda p: phase_force(p, t)
            current_v_dirichlet_bc = lambda p: velocity_dirichlet_bc(p, t)
            # Update phase-field using Allen-Cahn equation
            ac_A , ac_b = update_ac(u_n, phi_n, dt,current_phi_force, mesh_velocity)   # 此处差一个网格速度
            phi_val = spsolve(ac_A, ac_b,solver= 'scipy')
            phi[:] = phi_val[:-1]

            print("theta:",phi_val[-1])
            # Update auxiliary velocity field
            us_A , us_b = update_us(phi_n , phi , u_n ,s_n ,dt,
                                    current_v_force,current_v_dirichlet_bc,mesh_velocity)
            us[:] = spsolve(us_A, us_b,solver= 'scipy')
            # Update intermediate pressure
            ps_A , ps_b = update_ps(phi, us)
            ps[:] = spsolve(ps_A, ps_b,solver= 'scipy')
            # Update velocity field           
            u_A , u_b = update_velocity(phi, us, ps)
            u[:] = spsolve(u_A, u_b,solver= 'scipy')
            # Update gauge variable
            s_A , s_b = update_gauge(s_n, us)
            s[:] = spsolve(s_A, s_b,solver= 'scipy')
            # Update pressure field
            p_A , p_b = update_pressure(s, ps, dt)
            p[:] = spsolve(p_A, p_b,solver= 'scipy')
                        
            # Prepare for next time step
            phi_n[:] = phi[:]
            u_n[:] = u[:]
            s_n[:] = s[:]
            node_n = mesh.node.copy()
            mm.instance.uh = phi
            
            # Save results
            save_vtu(i+1,export_dir)
            # Compute and print bubble centroid
            centroid = compute_bubble_centroid()
            print(f"Time step {i+1}, Time {t:.4f}, Bubble Centroid: {centroid}")
            
        return u, u[:ugdof], u[ugdof:], p, phi