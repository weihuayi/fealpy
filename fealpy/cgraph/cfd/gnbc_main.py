from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

class GNBCSolver(CNodeType):
    r"""Two-Phase Couette Flow FEM Solver (GNBC Model)
    This node implements a finite element solver for a two-phase Couette flow
    governed by the coupled Cahn–Hilliard and Navier–Stokes equations
    under the Generalized Navier Boundary Condition (GNBC).
    It is designed to simulate interfacial slip and contact-angle effects
    in immiscible two-phase incompressible flows.

    The solver adopts a time-stepping algorithm using a uniform time line.
    At each time step, it sequentially solves:
        1. **Cahn–Hilliard (CH) equation** – updates the phase field `φ` and chemical potential `μ`;
        2. **Navier–Stokes (NS) equation** – updates the velocity field `u` and pressure field `p`.

    The implementation supports automatic boundary detection, Dirichlet condition application,
    and data export (in compressed JSON format). The results include the maximum and minimum
    x-velocity on the upper and lower walls, enabling slip-flow characterization.

    **Main Features**
    -----------------
    - Coupled CH–NS solver based on the GNBC model;
    - Finite element discretization with configurable function spaces (`phispace`, `uspace`, `pspace`);
    - Time-stepping integration with adaptive control via `UniformTimeLine`;
    - Boundary condition handling for wall slip, contact angles, and Dirichlet velocity;
    - Data export and compression for post-processing visualization;
    - Evaluation of wall slip velocity profiles.

    **Input Parameters**
    --------------------
    - `R`: Dimensionless system length scale  
    - `L_s`: Slip length (dimensionless)  
    - `epsilon`: Interface thickness (dimensionless)  
    - `L_d`: Mobility coefficient (dimensionless)  
    - `lam`: Double-well potential coefficient (dimensionless)  
    - `V_s`: Wall velocity (dimensionless)  
    - `s`: Stabilization parameter  
    - `theta_s`: Static contact angle (in radians)  
    - `h`: Mesh size  
    - `init_phi`: Initial phase field function  
    - `is_uy_Dirichlet`, `is_up_boundary`, `is_down_boundary`, `is_wall_boundary`: Boundary indicators  
    - `u_w`: Wall velocity boundary condition function  
    - `mesh`: Finite element mesh  
    - `phispace`, `space`, `pspace`, `uspace`: FE function spaces  
    - `output_dir`: Directory for data export  
    - `q`: Quadrature order (default: 5)

    **Output**
    ----------
    - `max_u_up`: Maximum x-velocity on the upper wall  
    - `min_u_up`: Minimum x-velocity on the upper wall  
    - `max_u_down`: Maximum x-velocity on the lower wall  
    - `min_u_down`: Minimum x-velocity on the lower wall  

    **Applications**
    ----------------
    - Two-phase incompressible Couette flow with wall slip;
    - Wetting and dewetting dynamics with contact angle effects;
    - Verification of GNBC boundary models in diffuse-interface formulations.
    """
    TITLE: str = "两相Couette流动问题模型"
    PATH: str = "流体.有限元算法"
    DESC: str = """
    该节点基于广义Navier边界条件(GNBC)实现两相Couette流动的有限元数值求解。
    模型将Cahn-Hilliard方程与Navier-Stokes方程进行耦合求解，
    通过时间步进迭代更新相场、化学势、速度和压力。
    
    - **主要功能：**
      1. 构建CH与NS方程的双线性与线性形式；
      2. 自动施加速度Dirichlet边界条件；
      3. 利用稀疏矩阵求解器(spsolve)高效求解线性系统；
      4. 支持时间步输出与数据压缩保存；
      5. 计算上、下壁面速度的最大/最小值以分析滑移特征。
    
    - **输入参数：**
      包含物理参数（R, L_s, ε, L_d, λ, V_s, s, θ_s 等）、
      初始条件与边界函数、有限元空间定义（phispace, uspace, pspace）、
      网格信息及输出目录等。
    
    - **输出结果：**
      返回上、下边界x方向速度的最大与最小值，
      用于评估滑移流动及速度分布。

    适用于两相不可压流体的层流模拟、界面滑移研究及润湿动力学分析。
"""
    INPUT_SLOTS = [
        PortConf("R", DataType.FLOAT, 1, title="系统长度尺度(无量纲)"),
        PortConf("L_s", DataType.FLOAT, 1, title = "物理滑移长度"),
        PortConf("epsilon", DataType.FLOAT, 1, title="界面厚度(无量纲)"),
        PortConf("L_d", DataType.FLOAT, 1, title="迁移系数(无量纲)"),
        PortConf("lam", DataType.FLOAT, 1, title="相场势能系数(无量纲)"),
        PortConf("V_s", DataType.FLOAT, 1, title="壁面速度(无量纲)"),
        PortConf("s", DataType.FLOAT, 1, title="稳定化参数(无量纲)"),
        PortConf("theta_s", DataType.TENSOR, 1, title="接触角(弧度)"),
        PortConf("h", DataType.FLOAT, 1, title="网格尺寸"),
        PortConf("init_phi", DataType.FUNCTION, 1, title="定义初始相场分布"),
        PortConf("is_uy_Dirichlet", DataType.FUNCTION, 1, title="判断是否为速度Dirichlet边界"),
        PortConf("is_up_boundary", DataType.FUNCTION, 1, title="判断是否为上边界"),
        PortConf("is_down_boundary", DataType.FUNCTION, 1, title="判断是否为下边界"),
        PortConf("is_wall_boundary", DataType.FUNCTION, 1, title="判断是否为壁面边界"),
        PortConf("u_w", DataType.FUNCTION, 1, title="定义壁面速度边界条件"),
        PortConf("mesh", DataType.MESH, 1, title="网格"),
        

        PortConf("phispace", DataType.SPACE, 1, title="相场函数空间"),
        PortConf("space", DataType.SPACE, 1, title="函数空间"),
        PortConf("pspace", DataType.SPACE, 1, title="压力函数空间"),
        PortConf("uspace", DataType.SPACE, 1, title="速度函数空间"),
        PortConf("output_dir", DataType.STRING, title="输出目录"),
        PortConf("q", DataType.INT, title="积分次数", default=5),
    ]
    OUTPUT_SLOTS = [
        PortConf("max_u_up", DataType.TENSOR, title="上边界最大值"),
        PortConf("min_u_up", DataType.TENSOR, title="上边界最小值"),
        PortConf("max_u_down", DataType.TENSOR, title="下边界最大值"),
        PortConf("min_u_down", DataType.TENSOR, title="下边界最小值")
    ]
    #pspace, phispace, space, uspace,
    @staticmethod
    def run(R ,L_s, epsilon, L_d, lam, V_s, s,
            theta_s,h,init_phi, is_uy_Dirichlet,is_up_boundary,
            is_down_boundary,is_wall_boundary,u_w,mesh,
            phispace,space, pspace,uspace,output_dir,q=5) -> Union[object]:
        from fealpy.backend import backend_manager as bm
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        from fealpy.old.timeintegratoralg import UniformTimeLine
        from fealpy.solver import spsolve, cg, gmres 
        from fealpy.fem import DirichletBC
        from fealpy.utils import timer
        import json
        import os
        import gzip
        from fealpy.cfd.example.GNBC.solver import Solver
        class PDE:
            def __init__(self, R, L_s, epsilon, L_d, lam, V_s, s,
                        theta_s, is_wall_boundary,is_up_boundary,
                        is_down_boundary,is_uy_Dirichlet, u_w, init_phi):
                self.R = R
                self.L_s = L_s
                self.epsilon = epsilon
                self.L_d = L_d
                self.lam = lam
                self.V_s = V_s
                self.s = s
                self.theta_s = theta_s
                self.is_wall_boundary = is_wall_boundary
                self.is_up_boundary = is_up_boundary
                self.is_down_boundary = is_down_boundary
                self.is_uy_Dirichlet = is_uy_Dirichlet
                self.u_w = u_w
                self.init_phi = init_phi
        
        
        T = 2
        nt = int(T/(0.1*h))
        pde = PDE(R, L_s, epsilon, L_d, lam, V_s, s,
            theta_s, is_wall_boundary,is_up_boundary,
            is_down_boundary,is_uy_Dirichlet,u_w,init_phi)
        
        timeline = UniformTimeLine(0, T, nt)
        dt = timeline.dt
        time = timer()
        next(time)
        solver = Solver(pde, mesh, pspace, phispace, uspace, dt, q)
        
        u0 = uspace.function()
        u1 = uspace.function()
        u2 = uspace.function()
        phi0 = phispace.interpolate(pde.init_phi)
        phi1 = phispace.function()
        # TODO:第一步求解
        phi1[:] = phi0[:]
        phi2 = phispace.function()
        mu1 = phispace.function()
        mu2 = phispace.function()
        p1 = pspace.function()
        p2 = pspace.function()

        ugdof = uspace.number_of_global_dofs()
        pgdof = pspace.number_of_global_dofs()
        phigdof = phispace.number_of_global_dofs()

        # fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
        mesh.nodedata['phi'] = phi0
        mesh.nodedata['u'] = u0.reshape(2,-1).T
        mesh.nodedata['mu'] = mu1
        # mesh.to_vtk(fname=fname)

        CH_BForm = solver.CH_BForm()
        CH_LForm = solver.CH_LForm()
        NS_BForm = solver.NS_BForm()
        NS_LForm = solver.NS_LForm()
        is_up = space.is_boundary_dof(pde.is_up_boundary)
        is_down = space.is_boundary_dof(pde.is_down_boundary)
        is_uy_bd = space.is_boundary_dof(pde.is_uy_Dirichlet)
        ux_gdof = space.number_of_global_dofs()
        is_bd = bm.concatenate((bm.zeros(ux_gdof, dtype=bool), is_uy_bd, bm.zeros(pgdof, dtype=bool)))
        NS_BC = DirichletBC(space=(uspace,pspace), \
                gd=bm.zeros(ugdof+pgdof, dtype=bm.float64), \
                threshold=is_bd, method='interp')
        node = mesh.interpolation_points(p=1)
        cell = mesh.entity('cell')
        time.send("初始化用时")
        data = []
        j = 0
        for i in range(nt):
            t = timeline.next_time_level()
            print(f"第{i+1}步")
            print("time=", t)
            solver.CH_update(u0, u1, phi0, phi1)
            CH_A = CH_BForm.assembly()
            CH_b = CH_LForm.assembly()
            
            time.send(f"第{i+1}次CH组装用时")
            CH_x = spsolve(CH_A, CH_b, 'mumps')
            time.send(f"第{i+1}次CH求解用时")
            
            phi2[:] = CH_x[:phigdof]
            mu2[:] = CH_x[phigdof:] 
            solver.NS_update(u0, u1, mu2, phi2, phi1)
            NS_A = NS_BForm.assembly()
            NS_b = NS_LForm.assembly()
            NS_A,NS_b = NS_BC.apply(NS_A,NS_b)
            time.send(f"第{i+1}次NS组装用时") 
            NS_x = spsolve(NS_A, NS_b, 'mumps') 
            time.send(f"第{i+1}次NS求解用时")
            u2[:] = NS_x[:ugdof]
            p2[:] = NS_x[ugdof:]
            
            u0[:] = u1[:]
            u1[:] = u2[:]
            phi0[:] = phi1[:]
            phi1[:] = phi2[:]
            mu1[:] = mu2[:]
            p1[:] = p2[:]
            
            # output = './'
            # fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
            mesh.nodedata['phi'] = phi2
            mesh.nodedata['u'] = u2.reshape(2,-1).T
            mesh.celldata['p'] = p2
            mesh.nodedata['mu'] = mu2
            # mesh.to_vtk(fname=fname)
            timeline.advance()
            time.send(f"第{i+1}次画图用时")
            uuu = u2.reshape(2,-1).T
            print("上边界最大值",bm.max(uuu[is_up,0]))
            print("上边界最小值",bm.min(uuu[is_up,0]))
            print("下边界最大值",bm.max(uuu[is_down,0]))
            print("下边界最小值",bm.min(uuu[is_down,0]))
            os.makedirs(output_dir, exist_ok=True)
            if i % 10 == 0:
                data.append ({
                    "time": round(i * dt, 8),
                    "值":{
                        "phi" : mesh.nodedata["phi"].tolist(),  # ndarray -> list
                        "u" : mesh.nodedata["u"].tolist(),  # ndarray -> list
                        "p" : mesh.celldata["p"].tolist(),  # ndarray -> list
                        "mu": mesh.nodedata["mu"].tolist(), # ndarray -> list
                    }, 
                     "几何": {
                    "cell": cell.tolist(),  # ndarray -> list
                    "node": node.tolist()   # ndarray -> list
                }
                    
                    })
                
                if len(data) == 10 :
                    j += 1
                    file_name = f"file_{j:08d}.json.gz"
                    file_path = os.path.join(output_dir, file_name)

                    with gzip.open(file_path, "wt", encoding="utf-8") as f:
                        json.dump(data, f, indent=4, ensure_ascii=False)
                    
                    data.clear()
        #next(time)
        max_u_up = bm.max(uuu[is_up,0])
        min_u_up = bm.min(uuu[is_up,0])
        max_u_down = bm.max(uuu[is_down,0])
        min_u_down = bm.min(uuu[is_down,0])
        return max_u_up, min_u_up, max_u_down, min_u_down
