from ..nodetype import CNodeType, PortConf, DataType


class DamBreak3DParticleGeneration(CNodeType):
    r"""3D Dam-Break Impact on Cylinder Particle Generator for SPH Simulations.

    This node generates the initial configuration settings for simulating the impact of 3D dam-break flow on a cylinder, 
    including the 3D spatial distribution of fluid particles, boundary particles, and cylinder particles.

    Fluid particles are concentrated in a cuboid region (simulating water body in a flume), while boundary particles 
    form a 3D container (consisting of the bottom and surrounding walls of the flume). Cylinder particles are vertically 
    placed in the fluid domain as a 3D cylindrical structure, which is used to simulate the impact effect of dam-break 
    flow on structural objects.

    By specifying the particle spacing (dx, dy, dz) in three spatial directions, the corresponding 3D particle distribution 
    can be generated, which can be further used for subsequent fluid-structure interaction analysis.

    Inputs:
        dx (float): Particle spacing in the X direction.
        dy (float): Particle spacing in the Y direction.
        dz (float): Particle spacing in the Z direction.

    Outputs:
        mesh (mesh): Spatial distribution of all particles (fluid, boundary, cylinder).
        box_size (tensor): Dimension parameters of the 3D simulation domain.
    """

    TITLE: str = "三维溃坝对柱影响问题粒子生成"
    PATH: str = "preprocess.mesher"
    DESC: str = (
        """这个节点用于生成三维溃坝对柱体影响问题模拟的初始设置，包括流体粒子、边界粒子、柱体粒子的三维空间分布。
    流体粒子集中在一个长方体区域（模拟水槽内的水体），边界粒子构成一个三维容器（底部与四周的墙壁）。
    柱体粒子以三维柱体形式竖直放置于流体域中，用于模拟溃坝水流对结构物的冲击作用。
    通过指定三个方向的粒子间距dx、dy和dz，可以生成相应的三维粒子分布，并可用于后续的流体‑结构相互作用分析。"""
    )
    INPUT_SLOTS = [
        PortConf("dx", dtype=DataType.FLOAT, ttype=1, title="X方向粒子间隔", default=0.02),
        PortConf("dy", dtype=DataType.FLOAT, ttype=1, title="Y方向粒子间隔", default=0.02),
        PortConf("dz", dtype=DataType.FLOAT, ttype=1, title="Z方向粒子间隔", default=0.02),
    ]
    OUTPUT_SLOTS = [
        PortConf("mesh", dtype=DataType.MESH, title="粒子分布"),
        PortConf("box_size", dtype=DataType.TENSOR, title="模拟区域"),
    ]

    @staticmethod
    def run(dx, dy, dz):
        from fealpy.backend import backend_manager as bm
        bm.set_backend("numpy")
        from fealpy.mesh.node_mesh import NodeMesh
        mesh = NodeMesh.from_dam_break_domain_3d(dx=dx,dy=dy,dz=dz)
        box_size = bm.array([1.6, 0.61, 0.6], dtype=bm.float64) #模拟区域
        return mesh, box_size

class SPHQueryDam(CNodeType):
    r"""Particle Neighbor Search for SPH Simulations.

    This node performs particle neighbor search operations for Smoothed Particle Hydrodynamics (SPH) simulations, 
    calculating neighbor information for each particle and relevant values of the kernel function (e.g., kernel value, gradient).

    Inputs:
        mesh (mesh): Spatial distribution of particles (the input particle set for neighbor search).
        kernel (string): Type of kernel function to be used. Available options: ["quintic", "cubic", "wendlandc2", "quintic_wendland"], default is "wendlandc2".
        h (float): Smoothing length of the kernel function (a key parameter for SPH kernel calculation).
        space (bool): Whether to enable periodic boundary conditions for the simulation space, default is False.
        box_size (tensor): Dimension parameters of the simulation domain (required for boundary condition calculation, especially periodic boundaries).

    Outputs:
        neighbors (tensor): Indices of neighboring particles corresponding to each target particle.
        self_node (tensor): Indices of the target particles themselves (one-to-one correspondence with the neighbors tensor).
        dr (tensor): Displacement difference vectors between each target particle and its neighboring particles.
        dist (tensor): Euclidean distances between each target particle and its neighboring particles.
        w (tensor): Kernel function values calculated for each particle-neighbor pair.
        grad_w (tensor): Gradient vectors of the kernel function for each particle-neighbor pair.
        grad_w_norm (tensor): Norm (magnitude) of the kernel function gradient vectors for each particle-neighbor pair.
    """

    TITLE: str = "粒子邻近搜索"
    PATH: str = "simulation.solvers"
    DESC: str = (
        """这个节点用于执行粒子邻近搜索操作，计算每个粒子的邻居信息和核函数相关值。"""
    )
    INPUT_SLOTS = [
        PortConf("mesh", dtype=DataType.MESH, title="粒子分布"),
        PortConf("kernel", DataType.STRING, 0, title="选取核函数",default="wendlandc2", items=["quintic", "cubic", "wendlandc2","quintic_wendland"]),
        PortConf("space", DataType.BOOL, 0, title="是否周期边界空间",default=False),
        PortConf("box_size", dtype=DataType.TENSOR, title="模拟区域"),
    ]
    OUTPUT_SLOTS = [
        PortConf("neighbors", dtype=DataType.TENSOR, title="邻居粒子索引"),
        PortConf("self_node", dtype=DataType.TENSOR, title="自身粒子索引"),
        PortConf("dr", dtype=DataType.TENSOR, title="粒子间位移差向量"),
        PortConf("dist", dtype=DataType.TENSOR, title="粒子间距离"),
        PortConf("w", dtype=DataType.TENSOR, title="核函数值"),
        PortConf("grad_w", dtype=DataType.TENSOR, title="梯度向量"),
        PortConf("grad_w_norm", dtype=DataType.TENSOR, title="梯度范数"),
    ]

    @staticmethod
    def run(mesh,kernel,space,box_size):
        from fealpy.cfd.simulation.sph.sph_base import SPHQueryKernel, Kernel
        kinfo = {
            "type": kernel,  # 选择核函数类型
            "h": mesh.nodedata["H"],           # 设定核函数的平滑长度
            "space": space      # 是否周期边界空间
        }
        kernel = Kernel(kinfo, dim=3)
        sph_query = SPHQueryKernel(
            mesh=mesh,
            radius=2 * mesh.nodedata["H"],
            box_size=box_size,
            mask_self=True,
            kernel_info=kinfo,
            periodic=[False, False, False],
            )
        self_node, neighbors = sph_query.find_node()
        dr = kernel.compute_displacement(self_node, neighbors, mesh.nodedata["position"], box_size)
        dist = kernel.compute_distance(self_node, neighbors, mesh.nodedata["position"], box_size)
        w = sph_query.compute_kernel_value(self_node, neighbors)
        grad_w, grad_w_norm = sph_query.compute_kernel_gradient(self_node, neighbors)
        return neighbors, self_node, dr, dist, w, grad_w, grad_w_norm 



class DamBreak3DParticleIterativeUpdate(CNodeType):
    r"""3D Dam-Break Impact on Cylinder Particle Iterative Update for SPH Simulations.

    This node implements a numerical simulation solver based on the Smoothed Particle Hydrodynamics (SPH) method 
    for the 3D dam-break impact on a cylinder scenario, focusing on the iterative update of particle properties 
    (e.g., velocity, pressure, spatial position) at each time step.

    Inputs:
        mesh (mesh): Initial/previous spatial distribution of particles (including fluid, boundary, and cylinder particles).
        i (int): Current time step index in the iterative simulation process.
        dt (float): Time step size for numerical integration, default is 0.001.
        neighbors (tensor): Indices of neighboring particles corresponding to each target particle (from neighbor search).
        self_node (tensor): Indices of the target particles themselves (one-to-one correspondence with the neighbors tensor).
        dr (tensor): Displacement difference vectors between each target particle and its neighboring particles.
        dist (tensor): Euclidean distances between each target particle and its neighboring particles.
        w (tensor): Kernel function values calculated for each particle-neighbor pair.
        grad_w (tensor): Gradient vectors of the kernel function for each particle-neighbor pair.
        output_dir (string): Directory path for saving simulation results (e.g., updated particle properties).

    Outputs:
        velocity (tensor): Updated velocity values of fluid particles after current time step iteration.
        pressure (tensor): Updated pressure values of fluid particles after current time step iteration.
        mesh (mesh): Updated spatial distribution of all particles (reflecting position changes after iteration).
    """

    TITLE: str = "三维溃坝对柱体影响场景粒子迭代更新"
    PATH: str = "simulation.solvers"
    DESC: str = """该节点实现了基于光滑粒子流体动力学(SPH)的三维溃坝对柱体影响场景数值模拟求解器，
    """
    INPUT_SLOTS = [
        PortConf("mesh", DataType.MESH, 1, title="粒子分布"),
        PortConf("i", DataType.INT, title="当前时间步"),
        PortConf("dt", DataType.FLOAT, 0, title="时间步长",default=0.001),
        PortConf("neighbors", dtype=DataType.TENSOR, title="邻居粒子索引"),
        PortConf("self_node", dtype=DataType.TENSOR, title="自身粒子索引"),
        PortConf("dr", dtype=DataType.TENSOR, title="粒子间位移差向量"),
        PortConf("dist", dtype=DataType.TENSOR, title="粒子间距离"),
        PortConf("w", dtype=DataType.TENSOR, title="核函数值"),
        PortConf("grad_w", dtype=DataType.TENSOR, title="梯度向量"),
        PortConf("output_dir", DataType.STRING, title="输出目录")
    ]
    OUTPUT_SLOTS = [
        PortConf("velocity", DataType.TENSOR, title="流体粒子速度"),
        PortConf("pressure", DataType.TENSOR, title="流体粒子压力"),
        PortConf("mesh", dtype=DataType.MESH, title="粒子分布"),
    ]
    

    @staticmethod
    def run(mesh, i, dt, neighbors, self_node, dr, dist, w, grad_w, output_dir):
        from fealpy.backend import backend_manager as bm
        from fealpy.cfd.simulation.sph.equation_solver import EquationSolver
        from fealpy.cfd.simulation.sph.particle_solver_new import SPHSolver
        from fealpy.cfd.simulation.utils_non_pyvista import VTKWriter2
        from pathlib import Path
        print(i)
        solver = EquationSolver()
        sphsolver = SPHSolver(mesh)
        if i % 30 == 1 and i != 1:
            sphsolver.rein_rho_3d(mesh.nodedata, self_node, neighbors, w)
            mesh.nodedata["mass"] = mesh.nodedata["rho"] * mesh.nodedata["dx"] * mesh.nodedata["dy"] * mesh.nodedata["dz"]
        
        mesh.nodedata["drhodt"] = sphsolver.change_rho_dam(mesh.nodedata, self_node, neighbors, grad_w)
        mesh.nodedata["rho"] += dt * mesh.nodedata["drhodt"]
        mesh.nodedata["rho"] = bm.maximum(mesh.nodedata["rho"], mesh.nodedata["rhomin"])
        
        mesh.nodedata["mass"] = mesh.nodedata["rho"] * mesh.nodedata["dx"] * mesh.nodedata["dy"] * mesh.nodedata["dz"]

        mesh.nodedata["pressure"] = solver.state_equation("tait_eos", mesh.nodedata, rho=mesh.nodedata["rho"], c0=mesh.nodedata["c0"], rho0=mesh.nodedata["rho0"],gamma=mesh.nodedata["gamma"])
        mesh.nodedata["sound"] = sphsolver.sound_dam(mesh.nodedata, mesh.nodedata["c0"], mesh.nodedata["rho0"], mesh.nodedata["gamma"])
        
        mesh.nodedata["dudt"] = sphsolver.change_u_dam(mesh.nodedata, self_node, neighbors, dr, dist, grad_w)
        mesh.nodedata["u"] = bm.where((mesh.nodedata["tag"] == 0)[:, None], mesh.nodedata["u"] + dt * mesh.nodedata["dudt"], mesh.nodedata["u"],)
        
        mesh.nodedata["dxdt"] = sphsolver.change_r_dam(mesh.nodedata, self_node, neighbors, w)
        mesh.nodedata["position"] = bm.where((mesh.nodedata["tag"] == 0)[:, None], mesh.nodedata["position"] + dt * mesh.nodedata["dxdt"], mesh.nodedata["position"],)

        current_data = {
            "position": mesh.nodedata["position"].tolist(),  # ndarray -> list
            "velocity": mesh.nodedata["u"].tolist(),  # ndarray -> list
            "pressure": mesh.nodedata["pressure"].tolist(),  # ndarray -> list
            }     
        writer = VTKWriter2()
        zfname = output_dir + '/' + 'test_'+ str(i+1).zfill(10) + '.vtk'    
        writer.write_vtk(current_data, zfname)           

        velocity = mesh.nodedata["u"]
        pressure = mesh.nodedata["pressure"]
        
        return velocity, pressure, mesh