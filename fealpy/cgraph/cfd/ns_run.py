
from ..nodetype import CNodeType, PortConf, DataType

def lagrange_multiplier(A, b, c=0, uspace=None, pspace=None):
    """
    Constructs the augmented system matrix for Lagrange multipliers.
    c is the integral of pressure, default is 0.
    """
    from fealpy.sparse import COOTensor
    from fealpy.backend import backend_manager as bm
    from fealpy.fem import SourceIntegrator, LinearForm
    from fealpy.fem import BlockForm
    LagLinearForm = LinearForm(pspace)
    LagLinearForm.add_integrator(SourceIntegrator(source=1))
    LagA = LagLinearForm.assembly()
    LagA = bm.concatenate([bm.zeros(uspace.number_of_global_dofs()), LagA], axis=0)

    A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                            bm.arange(len(LagA), dtype=bm.int32)]), LagA, spshape=(1, len(LagA)))

    A = BlockForm([[A, A1.T], [A1, None]])
    A = A.assembly_sparse_matrix(format='csr')
    b0 = bm.array([c])
    b  = bm.concatenate([b, b0], axis=0)
    return A, b

__all__ = ["StationaryNSRun"]

class StationaryNSRun(CNodeType):
    r"""Finite element iterative solver for steady incompressible Navier-Stokes equations.

    Inputs:
        maxstep (int): Maximum number of nonlinear iterations.
        tol (float): Convergence tolerance based on velocity and pressure residuals.
        update (function): Function to update coefficients or nonlinear terms.
        apply_bc (function): Function to apply Dirichlet boundary conditions.
        BForm (linops): Bilinear form operator for system matrix assembly.
        LForm (linops): Linear form operator for right-hand side vector assembly.
        uspace(space): Velocity function space.
        pspace(space): Pressure function space.
        mesh(mesh): Computational mesh.

    Outputs:
        uh (tensor): Final numerical velocity field.
        uh_x (tensor): x-component of the velocity field.
        uh_y (tensor): y-component of the velocity field.
        ph (tensor): Final numerical pressure field.
    """
    TITLE: str = "稳态 NS 方程有限元迭代求解"
    PATH: str = "流体.NS 方程有限元迭代求解"
    DESC: str = """该节点实现稳态不可压 Navier-Stokes 方程的有限元迭代求解器，通过系数更新与边界
                条件施加，逐步组装并求解线性系统，输出速度场与压力场的稳态数值解。"""
    INPUT_SLOTS = [
        PortConf("maxstep", DataType.INT, 0, default=1000, min_val=1, title="最大迭代步数"),
        PortConf("tol", DataType.FLOAT, 0, default=1e-6, min_val=1e-12, max_val=1e-2, title="残差"),
        PortConf("update", DataType.FUNCTION, title="更新函数"),
        PortConf("apply_bc", DataType.FUNCTION, title="边界处理函数"),
        PortConf("BForm", DataType.LINOPS, title="算子"),
        PortConf("LForm", DataType.LINOPS, title="向量"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR, title="速度数值解"),
        PortConf("uh_x", DataType.TENSOR, title="速度x分量数值解"),
        PortConf("uh_y", DataType.TENSOR, title="速度y分量数值解"),
        PortConf("ph", DataType.TENSOR, title="压力数值解")
    ]
    @staticmethod
    def run(maxstep, tol, update, apply_bc, BForm, LForm, uspace, pspace, mesh):
        from fealpy.solver import spsolve
        uh0 = uspace.function()
        ph0 = pspace.function()
        uh1 = uspace.function()
        ph1 = pspace.function()
        ugdof = uspace.number_of_global_dofs()
        for i in range(maxstep):
            update(uh0)
            A = BForm.assembly()
            F = LForm.assembly()
            A, F = apply_bc(A, F)
            A, F = lagrange_multiplier(A, F, c = 0, uspace=uspace, pspace=pspace)
            x = spsolve(A, F,"mumps")
            uh1[:] = x[:ugdof]
            ph1[:] = x[ugdof:-1]
            res_u = mesh.error(uh0, uh1)
            res_p = mesh.error(ph0, ph1)
            
            if res_u + res_p < tol:
                break
            uh0[:] = uh1
            ph0[:] = ph1

        NN = mesh.number_of_nodes()
        uh_x = uh1[:int(ugdof/2)]
        uh_x = uh_x[:NN]
        uh_y = uh1[int(ugdof/2):]
        uh_y = uh_y[:NN]

        return uh1, uh_x, uh_y, ph1
    
class IncompressibleNSIPCSRun(CNodeType):
    r"""IPCS iterative solver for unsteady incompressible Navier-Stokes equations.

    Inputs:
        T0 (float): Initial time.
        T1 (float): Final time.
        NL (int): Number of time levels.
        uspace(space): Velocity function space.
        pspace(space): Pressure function space.
        velocity_0 (function): Initial velocity field.
        pressure_0 (function): Initial pressure field.
        is_pressure_boundary (function): Predicate function for pressure boundary regions.
        predict_velocity (function): Function to assemble predicted velocity system.
        correct_pressure (function): Function to assemble pressure correction system.
        correct_velocity (function): Function to assemble velocity correction system.
        mesh(mesh): Computational mesh. 
    Outputs:
        uh (tensor): Final numerical velocity field.
        ph (tensor): Final numerical pressure field.
        uh_x (tensor): x-component of the velocity field.
        uh_y (tensor): y-component of the velocity field.
    """
    TITLE: str = "IPCS 求解非稳态 NS 方程"
    PATH: str = "流体.NS 方程有限元迭代求解"
    DESC: str  = """该节点实现非稳态不可压 Navier-Stokes 方程的 IPCS 分步算法求解器，按时间步推进依次完成速度预测、
                压力修正与速度校正，并输出速度与压力场的时序数值结果。"""
    INPUT_SLOTS = [
        PortConf("T0", DataType.FLOAT, title="初始时间"),
        PortConf("T1", DataType.FLOAT, title="结束时间"),
        PortConf("NL", DataType.INT, title="时间层数"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("velocity_0", DataType.FUNCTION, title="初始速度"),
        PortConf("pressure_0", DataType.FUNCTION, title="初始压力"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界"),
        PortConf("predict_velocity", DataType.FUNCTION, title="预测速度方程离散"),
        PortConf("correct_pressure", DataType.FUNCTION, title="压力修正方程离散"),
        PortConf("correct_velocity", DataType.FUNCTION, title="速度修正方程离散"),
        PortConf("mesh", DataType.MESH, title="网格"),
        PortConf("output_dir", DataType.STRING, title="输出目录")
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR, title="速度数值解"),
        PortConf("ph", DataType.TENSOR, title="压力数值解"),
        PortConf("uh_x", DataType.TENSOR, title="速度x分量数值解"),
        PortConf("uh_y", DataType.TENSOR, title="速度y分量数值解"),
        PortConf("uh_z", DataType.TENSOR, title="速度z分量数值解")
    ]
    def run(T0, T1, NL, uspace, pspace, velocity_0, pressure_0, is_pressure_boundary,
            predict_velocity, correct_pressure, correct_velocity, mesh, output_dir):
        from fealpy.backend import backend_manager as bm
        from fealpy.decorator import cartesian
        from fealpy.solver import spsolve
        from fealpy.cfd.simulation.time import UniformTimeLine
        import json
        import os
        import gzip
        
        nt = NL - 1
        timeline = UniformTimeLine(T0, T1, nt)
        dt = timeline.dt
        u0 = uspace.interpolate(cartesian(lambda p: velocity_0(p, timeline.T0)))
        p0 = pspace.interpolate(cartesian(lambda p: pressure_0(p, timeline.T0)))
        ugdof = uspace.number_of_global_dofs()
        pgdof = pspace.number_of_global_dofs()
        NN = mesh.number_of_nodes()

        node = mesh.interpolation_points(p=1)
        cell = mesh.entity('cell')
        data = []
        j = 0

        for i in range(nt):
            t  = timeline.current_time()
            # print(f"time={t}")
            
            uh1 = u0.space.function()
            uhs = u0.space.function()
            ph1 = p0.space.function()
             
            A0, b0 = predict_velocity(u0, p0, t = timeline.next_time(), dt = dt)
            uhs[:] = spsolve(A0, b0)

            A1, b1 = correct_pressure(uhs, p0, t = timeline.next_time(), dt = dt)
            if is_pressure_boundary() == 0:
                ph1[:] = spsolve(A1, b1)[:-1]
            else:
                ph1[:] = spsolve(A1, b1)

            A2, b2 = correct_velocity(uhs, p0, ph1, t = timeline.next_time(), dt = dt)
            uh1[:] = spsolve(A2, b2)

            u1 = uh1 
            p1 = ph1
            
            u0[:] = u1
            p0[:] = p1

            uh = u1
            uh = uh.reshape(mesh.GD,-1).T
            uh = uh[:NN,:]
            uh_x = uh[..., 0]
            uh_y = uh[..., 1]
            if mesh.GD == 3:
                uh_z = uh[..., 2]
            else:
                uh_z = bm.zeros_like(uh_x)
            ph = p1[:NN]

            # mesh.nodedata['ph'] = ph
            # mesh.nodedata['uh'] = uh.reshape(mesh.GD,-1).T
            # mesh.to_vtk(f'ns2d_{str(i+1).zfill(10)}.vtu')

            if (i+1) % 10 == 0:

                os.makedirs(output_dir, exist_ok=True)  # 创建目录

                data.append ({
                "time": round(t+dt, 8),
                "值":{
                    "uh" : uh.tolist(),  # ndarray -> list
                    "uh_x" : uh_x.tolist(),  # ndarray -> list
                    "uh_y" : uh_y.tolist(),  # ndarray -> list
                    "uh_z" : uh_z.tolist(),  # ndarray -> list
                    "ph" : ph.tolist()
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
            
            timeline.advance()

        return uh, ph, uh_x, uh_y, uh_z
