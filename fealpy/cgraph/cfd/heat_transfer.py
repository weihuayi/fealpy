from ..nodetype import CNodeType, PortConf, DataType


class HeatTransferParticleGeneration(CNodeType):
    r"""Heat Transfer particle generator for SPH simulations.

    This node generates initial particle configurations for heat transfer simulations
    using the Smoothed Particle Hydrodynamics (SPH) method.

    Inputs:
        dx (float): Horizontal spacing between particles.
        dy (float): Vertical spacing between particles.

    Outputs:
        mesh (mesh): Particle distribution in flat plate heat transfer.
    """

    TITLE: str = "平板传热问题粒子生成"
    PATH: str = "preprocess.mesher"
    DESC: str = (
        """这个节点用于生成平板传热模拟的初始设置，包括流体粒子和边界粒子的位置。
        流体粒子集中在一个矩形区域（模拟平板内的水），边界粒子构成一个容器（底部、左侧和右侧的墙壁）。
        通过指定两个方向粒子间距dx和dy，可以生成相应的粒子分布。"""
    )
    INPUT_SLOTS = [
        PortConf("dx", dtype=DataType.FLOAT, ttype=1, title="水平粒子间隔", default=0.02),
        PortConf("dy", dtype=DataType.FLOAT, ttype=1, title="垂直粒子间隔", default=0.02),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", dtype=DataType.MESH, title="粒子分布"),
        PortConf("box_size", dtype=DataType.TENSOR, title="模拟区域"),
    ]

    @staticmethod
    def run(dx, dy):
        from fealpy.backend import backend_manager as bm
        bm.set_backend("pytorch")
        from fealpy.mesh.node_mesh import NodeMesh
        mesh = NodeMesh.from_heat_transfer_domain(dx=dx,dy=dy)
        box_size = bm.array([1.0, 0.2 + dx*3*2], dtype=bm.float64) #模拟区域
        return mesh, box_size

class SPHQuery(CNodeType):
    r"""Heat Transfer particle generator for SPH simulations.

    This node generates initial particle configurations for heat transfer simulations
    using the Smoothed Particle Hydrodynamics (SPH) method.

    Inputs:
        dx (float): Horizontal spacing between particles.
        dy (float): Vertical spacing between particles.

    Outputs:
        mesh (mesh): Particle distribution in flat plate heat transfer.
    """

    TITLE: str = "粒子临近搜索"
    PATH: str = "simulation.solvers"
    DESC: str = (
        """这个节点用于执行粒子临近搜索操作，计算每个粒子的邻居信息和核函数相关值。"""
    )
    INPUT_SLOTS = [
        PortConf("mesh", dtype=DataType.MESH, title="粒子分布"),
        PortConf("kernel", DataType.STRING, 0, title="选取核函数",default="quintic", items=["quintic", "cubic", "wendlandc2","quintic_wendland"]),
        PortConf("dx", DataType.FLOAT, 0, title="核函数的平滑长度",default=0.02),
        PortConf("space", DataType.BOOL, 0, title="是否周期边界空间",default=True),
        PortConf("box_size", dtype=DataType.TENSOR, title="模拟区域"),
    ]
    OUTPUT_SLOTS = [
        PortConf("neighbors", dtype=DataType.TENSOR, title="当前粒子"),
        PortConf("self_node", dtype=DataType.TENSOR, title="临近粒子"),
        PortConf("dr", dtype=DataType.TENSOR, title="位移差向量"),
        PortConf("dist", dtype=DataType.TENSOR, title="节点间距离"),
        PortConf("w", dtype=DataType.TENSOR, title="核函数值"),
        PortConf("grad_w", dtype=DataType.TENSOR, title="梯度向量"),
        PortConf("grad_w_norm", dtype=DataType.TENSOR, title="梯度范数"),
        
    ]

    @staticmethod
    def run(mesh,kernel,dx,space,box_size):
        from fealpy.cfd.simulation.sph.sph_base import SPHQueryKernel, Kernel
        kinfo = {
            "type": kernel,  # 选择核函数类型
            "h": dx,           # 设定核函数的平滑长度
            "space": space      # 是否周期边界空间
        }
        kernel = Kernel(kinfo, dim=2)
        print(type(mesh))
        sph_query = SPHQueryKernel(
            mesh=mesh,
            radius=3*dx,
            box_size=box_size,
            mask_self=True,
            kernel_info=kinfo,
            periodic=[True, True, True],
            )
        self_node, neighbors = sph_query.find_node()
        dr = kernel.compute_displacement(self_node, neighbors, mesh.nodedata["position"], box_size)
        dist = kernel.compute_distance(self_node, neighbors, mesh.nodedata["position"], box_size)
        w = sph_query.compute_kernel_value(self_node, neighbors)
        grad_w, grad_w_norm = sph_query.compute_kernel_gradient(self_node, neighbors)
        return neighbors, self_node, dr, dist, w, grad_w, grad_w_norm 



class HeatTransferParticleIterativeUpdate(CNodeType):
    r"""Heat Transfer particle iterative update node for SPH simulations.

    This node implements the time-stepping update process for heat transfer simulations
    based on the Smoothed Particle Hydrodynamics (SPH) method. It performs iterative
    updates of fluid particle quantities such as velocity, pressure, and temperature
    over multiple time steps, simulating transient heat transfer in a flat plate scenario.

    The update follows the SPH formulation, including density summation, pressure
    computation via Tait equation of state, momentum and heat equations, and enforcement
    of boundary conditions. The simulation data are periodically written to compressed
    JSON files for postprocessing.

    Inputs:
        mesh (mesh): Initial particle distribution for SPH simulation.
        maxstep (int): Maximum number of iteration steps.
        kernel (str): Type of SPH kernel function to use.
            Available options: ["quintic", "cubic", "wendlandc2", "quintic_wendland"].
        dx (float): Smoothing length for the kernel function.
        dt (float): Time step size for the simulation.
        output_dir (str): Directory where output files will be saved.

    Outputs:
        velocity (Tensor): Particle velocities after simulation.
        pressure (Tensor): Particle pressures after simulation.
        temperature (Tensor): Particle temperatures after simulation.
    """

    TITLE: str = "平板传热场景粒子迭代更新"
    PATH: str = "simulation.solvers"
    DESC: str = """该节点实现了基于光滑粒子流体动力学(SPH)的平板传热场景数值模拟求解器，
    """
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, title="粒子分布"),
        PortConf("i", DataType.INT, title="当前时间步"),
        PortConf("box_size", dtype=DataType.TENSOR, title="模拟区域"),
        PortConf("dx", DataType.FLOAT, 0, title="核函数的平滑长度",default=0.02),
        PortConf("dt", DataType.FLOAT, 0, title="时间步长",default=0.00045454545454545455),
        PortConf("neighbors", dtype=DataType.TENSOR, title="当前粒子"),
        PortConf("self_node", dtype=DataType.TENSOR, title="临近粒子"),
        PortConf("dr", dtype=DataType.TENSOR, title="位移差向量"),
        PortConf("dist", dtype=DataType.TENSOR, title="节点间距离"),
        PortConf("w", dtype=DataType.TENSOR, title="核函数值"),
        PortConf("grad_w", dtype=DataType.TENSOR, title="梯度向量"),
        PortConf("grad_w_norm", dtype=DataType.TENSOR, title="梯度范数"),
        PortConf("output_dir", DataType.STRING, title="输出目录")
    ]
    OUTPUT_SLOTS = [
        PortConf("velocity", DataType.TENSOR, title="流体粒子速度"),
        PortConf("pressure", DataType.TENSOR, title="流体粒子压力"),
        PortConf("temperature", DataType.TENSOR, title="流体粒子温度"),
        PortConf("mesh", dtype=DataType.MESH, title="粒子分布"),
    ]
    

    @staticmethod
    def run(mesh, i, box_size, dx, dt,neighbors, self_node, 
            dr, dist, w, grad_w, grad_w_norm ,output_dir):
        from fealpy.backend import backend_manager as bm
        from fealpy.mesh.node_mesh import Space
        from fealpy.cfd.simulation.sph.equation_solver import EquationSolver
        from fealpy.cfd.simulation.sph.processing_technology import ProcessingTechnology
        from fealpy.cfd.simulation.utils_non_pyvista import VTKWriter2
        from pathlib import Path
        print(i)
        solver = EquationSolver()
        tech = ProcessingTechnology(mesh)
        space = Space()
        _, shift = space.periodic(side=box_size)

        if i == 0:
            mesh.nodedata["p"] = solver.state_equation("tait_eos", mesh.nodedata, X=5.0)
            mesh.nodedata = tech.boundary_conditions(mesh.nodedata, box_size, dx=dx)
        
        mesh.nodedata['mv'] += 1.0*dt*mesh.nodedata["dmvdt"]
        mesh.nodedata['tv'] = mesh.nodedata['mv']
        mesh.nodedata["position"] = shift(mesh.nodedata["position"], 1.0 * dt * mesh.nodedata["tv"])
            
        
        g_ext = tech.external_acceleration(mesh.nodedata["position"], box_size, dx=dx)
        fluid_mask = bm.where(mesh.nodedata["tag"] == 0, 1.0, 0.0) > 0.5
        rho_summation = solver.mass_equation_solve(0, mesh.nodedata, neighbors, w)
        rho = bm.where(fluid_mask, rho_summation, mesh.nodedata["rho"])
        
        p = solver.state_equation("tait_eos", mesh.nodedata, rho=rho, X=5.0)
        pb = solver.state_equation("tait_eos", mesh.nodedata, rho=bm.zeros_like(p), X=5.0)
        p, rho, mv, tv, T = tech.enforce_wall_boundary(mesh.nodedata, p, g_ext, neighbors, self_node, w, dr, with_temperature=True)
        mesh.nodedata["rho"] = rho
        mesh.nodedata["mv"] = mv
        mesh.nodedata["tv"] = tv
        
        T += dt * mesh.nodedata["dTdt"]
        mesh.nodedata["T"] = T
        mesh.nodedata["dTdt"] = solver.heat_equation_solve(0, mesh.nodedata, dr, dist, neighbors, self_node, grad_w)
        mesh.nodedata["dmvdt"] = solver.momentum_equation_solve(0,\
            mesh.nodedata, neighbors, self_node, dr, dist, grad_w_norm, p)
        mesh.nodedata["dmvdt"] = mesh.nodedata["dmvdt"] + g_ext
        mesh.nodedata["p"] = p
        mesh.nodedata["dtvdt"] = solver.momentum_equation_solve(1,\
            mesh.nodedata, neighbors, self_node, dr, dist, grad_w_norm, pb)
        mesh.nodedata = tech.boundary_conditions(mesh.nodedata, box_size, dx=dx)
        
        
        writer = VTKWriter2()
        export_dir = Path(output_dir).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        fname = export_dir / f"test_{str(i+1).zfill(10)}.vtk"
        writer.write_vtk(mesh.nodedata, fname)           

        velocity = mesh.nodedata["mv"]
        pressure = mesh.nodedata["p"]
        temperature = mesh.nodedata["T"]
        return velocity, pressure, temperature, mesh