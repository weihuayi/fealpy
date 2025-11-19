
from ..nodetype import CNodeType, PortConf, DataType

class CHNSFEMRun(CNodeType):
    r"""Finite Element Solver for the Coupled Cahn–Hilliard–Navier–Stokes (CHNS) Equations.

    This node implements a time-dependent finite element solver for the two-phase incompressible flow
    described by the **Cahn–Hilliard–Navier–Stokes (CHNS)** equations. The CHNS model captures 
    interface dynamics between immiscible fluids using a phase-field formulation coupled with
    Navier–Stokes equations.

    The solver advances the system in time using a given time step (`dt`) and total number of steps (`nt`).
    It alternates between solving:
        - **Cahn–Hilliard (CH)** equation: updates the phase field (`phi`) and chemical potential (`mu`);
        - **Navier–Stokes (NS)** equation: updates the velocity (`u`) and pressure (`p`) fields,
          with the local density `rho(phi)` determined by the phase field.

    Inputs:
        dt (float): Time step size.
        nt (int): Number of time steps.
        rho_up (float): Density of the upper fluid.
        rho_down (float): Density of the lower fluid.
        Fr (float): Froude number (controls gravitational effects).
        ns_update (function): Function that assembles the Navier–Stokes system.
        ch_update (function): Function that assembles the Cahn–Hilliard system.
        phispace (SpaceType): Finite element function space for the phase field.
        uspace (SpaceType): Function space for the velocity field.
        pspace (SpaceType): Function space for the pressure field.
        is_ux_boundary (function): Predicate for x-velocity component boundary.
        is_uy_boundary (function): Predicate for y-velocity component boundary.
        init_interface (function): Initial phase-field (interface) function.
        mesh (MeshType): Finite element mesh.

    Outputs:
        u (Function): Velocity vector field at the final time.
        ux (Function): x-component of velocity field.
        uy (Function): y-component of velocity field.
        p (Function): Pressure field.
        phi (Function): Phase-field function.
    """
    TITLE: str = "有限元求解 CHNS 方程"
    PATH: str = "流体.CHNS 方程有限元求解"
    DESC: str = """该节点实现了两相不可压流体的 Cahn–Hilliard–Navier–Stokes (CHNS) 方程组有限元求解
                器。CHNS 模型结合了相场法与流体力学方程，用以描述两种不可混溶流体的界面演化与流动耦合过程。
                通过时间步推进（dt, nt），程序在每个时间步内依次执行：
                1. Cahn–Hilliard 方程：更新相场函数 φ 与化学势 μ；
                2. Navier–Stokes 方程：根据当前相场计算密度场 ρ(φ)，更新速度场 u 与压力场 p。

                输入参数：
                - dt ：时间步长；
                - nt ：总时间步数；
                - rho_up 、rho_down ：上下两种流体的密度；
                - Fr ：弗劳德数，用于控制重力源项；
                - ns_update ：组装 NS 方程离散系统的函数；
                - ch_update ：组装 CH 方程离散系统的函数；
                - phispace、uspace、pspace ：分别为相场、速度与压力的有限元空间；
                - is_ux_boundary、is_uy_boundary ：定义速度边界条件的判定函数；
                - init_interface ：初始界面函数；
                - mesh ：计算区域网格。

                输出结果：
                - u ：最终时刻的速度场；
                - ux、uy ：速度在 x、y 方向的分量；
                - p ：压力场；
                - phi ：相场函数。

                使用示例：
                可将“相场更新模块 (CH)”与“流体更新模块 (NS)”分别连接到 `ch_update` 与 `ns_update`，
                设置好初始界面与网格信息后，即可在多时间步迭代中自动完成流体界面的演化与速度场的时序求解。
                """
    INPUT_SLOTS = [
        PortConf("dt", DataType.FLOAT, 0, title="时间步长", default=0.001),
        PortConf("nt", DataType.INT, 0, title="时间步数", default=2000),
        PortConf("rho_up", DataType.FLOAT, title="上层流体密度"),
        PortConf("rho_down", DataType.FLOAT, title="下层流体密度"),
        PortConf("Fr", DataType.FLOAT, title="弗劳德数"),
        PortConf("ns_update", DataType.FUNCTION, title="NS 更新函数"),
        PortConf("ch_update", DataType.FUNCTION, title="CH 更新函数"),
        PortConf("phispace", DataType.SPACE, title="相场函数空间"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("is_ux_boundary", DataType.FUNCTION, title="速度 x 分量边界"),
        PortConf("is_uy_boundary", DataType.FUNCTION, title="速度 y 分量边界"),
        PortConf("init_interface", DataType.FUNCTION, title="初始界面函数"),
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("output_dir", DataType.STRING, title="输出目录")
    ]
    OUTPUT_SLOTS = [
        PortConf("u", DataType.FUNCTION, title="速度场"),
        PortConf("p", DataType.FUNCTION, title="压力场"),
        PortConf("phi", DataType.FUNCTION, title="相场函数")
    ]
    @staticmethod
    def run(dt, nt, rho_up, rho_down, Fr, ns_update, ch_update,phispace, 
            uspace, pspace, is_ux_boundary, is_uy_boundary, init_interface, 
            mesh, output_dir):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import barycentric
        from fealpy.solver import spsolve
        from fealpy.fem import DirichletBC
        from pathlib import Path
        import time
        import os
        
        def set_rho(phi, rho_up, rho_down):
            result = phi.space.function()
            result[:] = (rho_up - rho_down)/2 * phi[:]
            result[:] += (rho_up + rho_down)/2 
            return result

        phigdof = phispace.number_of_global_dofs()
        phi0 = phispace.interpolate(init_interface)
        phi1 = phispace.interpolate(init_interface)
        phi2 = phispace.function()
        mu1 = phispace.function()
        mu2 = phispace.function()

        ugdof = uspace.number_of_global_dofs()
        pgdof = pspace.number_of_global_dofs()
        u0 = uspace.function()
        u1 = uspace.function()
        u2 = uspace.function()
        p1 = pspace.function()
        p2 = pspace.function()
        export_dir = Path(output_dir).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        mesh.nodedata["uh"] = u2.reshape(mesh.GD,-1).T
        mesh.nodedata["ph"] = p2
        mesh.nodedata["phih"] = phi2
        fname = export_dir / f"test_{str(0).zfill(10)}.vtu"
        mesh.to_vtk(fname=str(fname))


        is_bd = uspace.is_boundary_dof((is_ux_boundary, is_uy_boundary), method='interp')
        is_bd = bm.concatenate((is_bd, bm.zeros(pgdof, dtype=bm.bool)))
        gd = bm.concatenate((bm.zeros(ugdof, dtype=bm.float64), bm.zeros(pgdof, dtype=bm.float64)))
        BC = DirichletBC((uspace, pspace), gd=gd, threshold=is_bd, method='interp')

        node = mesh.entity_barycenter('node')
        tol = 1e-14
        left_bd = bm.where(bm.abs(node[:, 0]) < tol)[0]
        right_bd = bm.where(bm.abs(node[:, 0]-1.0) < tol)[0]


        for i in range(nt):
            # 设置参数
            print("iteration:", i)
            
            t0 = time.time()
            
            ch_A, ch_b = ch_update(u0, u1, phi0, phi1, dt)
            ch_A = ch_A.assembly()
            ch_b = ch_b.assembly()
            t1 = time.time()
            ch_x = spsolve(ch_A, ch_b, 'mumps')
            t2 = time.time()

            phi2[:] = ch_x[:phigdof]
            mu2[:] = ch_x[phigdof:]  
            
            # 更新NS方程参数
            t3 = time.time()

            rho = set_rho(phi1, rho_up, rho_down) 
            @barycentric
            def body_force(bcs, index):
                result = rho(bcs, index)
                result = bm.stack((result, result), axis=-1)
                result[..., 0] = (1/Fr) * result[..., 0] * 0
                result[..., 1] = (1/Fr) * result[..., 1] * -1
                return result
            
            ns_A, ns_b = ns_update(u0, u1, dt, rho, body_force)
            ns_A = ns_A.assembly()
            ns_b = ns_b.assembly()
            ns_A,ns_b = BC.apply(ns_A, ns_b)
            t4 = time.time() 
            ns_x = spsolve(ns_A, ns_b, 'mumps')
            t5 = time.time()

            print("CH组装时间:", t1-t0)
            print("求解CH方程时间:", t2-t1)
            print("NS组装时间:", t4-t3)
            print("求解NS方程时间:", t5-t4)
            u2[:] = ns_x[:ugdof]
            p2[:] = ns_x[ugdof:]
                
            u0[:] = u1[:]
            u1[:] = u2[:]
            phi0[:] = phi1[:]
            phi1[:] = phi2[:]
            mu1[:] = mu2[:]
            p1[:] = p2[:]

            phi2_lbdval = phi2[left_bd]
            mask = bm.abs(phi2_lbdval) < 0.5
            index = left_bd[mask]
            left_point = node[index, :]
            print("界面与左边界交点:", left_point)

            phi2_rbdval = phi2[right_bd]
            mask = bm.abs(phi2_rbdval) < 0.5
            index = right_bd[mask]
            right_point = node[index, :]
            print("界面与右边界交点:", right_point)

            mesh.nodedata["uh"] = u2
            mesh.nodedata["ph"] = p2
            mesh.nodedata["phih"] = phi2
            fname = export_dir / f"test_{str(i+1).zfill(10)}.vtu"
            mesh.to_vtk(fname=str(fname))

        return u2, p2, phi2