from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike


class HydraulicPipeFSIFEMModel:
    """
    """
    def __init__(self, options: dict, pde):
        self.options = options
        self.options["pde"] = pde
        self.max_iter = options["max_iter"]  # 最大迭代次数
        self.tolerance = options["tolerance"]  # 收敛容忍度
        self.pde = pde

    def run(self):
        """
        执行流固耦合算法
        """
        from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
        from fealpy.cfd.simulation.fem.stationary_incompressible_ns import Ossen, Newton
        from fealpy.cfd import StationaryIncompressibleNSLFEMModel
        from fealpy.solver import spsolve
        from fealpy.mesh import TriangleMesh
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        import time

        for i in range(self.max_iter):
            print(i+1)
            # 1. 求解流体方程，计算压力
            pde = self.pde
            fluid_model = StationaryIncompressibleNSLFEMModel(pde=pde, mesh=pde.fluid_mesh, options=self.options)
            fluid_model.equation.set_coefficient('convection', pde.fluid_rho)
            fluid_model.equation.set_coefficient('viscosity', pde.mu)
            u1, p1 = fluid_model.run()
            pde.fluid_mesh.nodedata["u"] = u1.reshape(3, -1).T
            pde.fluid_mesh.nodedata["p"] = p1

            # # 2. 压力传递
            # from .coupling_interface import CouplingInterface
            # interface = CouplingInterface(pde=pde)
            # p_interface = interface.pressure_on_interface(p1 = p1)
            # p_solid = interface.pressure_on_solid(p_interface)
            # interface_mesh = pde.interface_mesh
            # interface_mesh.nodedata["p"] = p_interface.reshape(3, -1).T

            # # 3. 求解固体方程，计算位移
            # from fealpy.csm.fem.hydraulic_pipe_lfem_model import  HydraulicPipeLFEMModel
            # from fealpy.decorator import barycentric
            # model = HydraulicPipeLFEMModel(self.options)
            # model.set_pde(pde)
            # A, F = model.linear_system()
            # @barycentric
            # def NeumannSI_source(bcs, index):
            #     result = -p_solid(bcs, index)
            #     return result
            # model.NeumannSI.source = NeumannSI_source
            # A = A.assembly()
            # F = F.assembly()
            # A1, F1 = model.apply_bc(A, F)
            # x = model.solve(A1, F1)
            # uh = model.space.function()
            # uh[:] = x
            # print("max displacement:", float(bm.max(bm.abs(uh))))
            # print(float(bm.linalg.norm(uh)))
            # pde.solid_mesh.nodedata["u"] = uh.reshape(-1, 3)
            # print("-----------------------------")

            # # 4. 检查收敛性（可以使用位移变化、压力变化等作为标准）
            # stress = interface.structural_stress_on_interface(uh)
            # pde.interface_mesh.nodedata["structural_stress"] = stress.reshape(3, -1).T

            # # 5. 网格更新
            # from fealpy.mesh import TetrahedronMesh, TriangleMesh
            # pde = self.pde
            # is_wall = pde.is_wall_boundary(pde.solid_mesh.entity('node'))
            # space = LagrangeFESpace(mesh=pde.interface_mesh, p=1)
            # solid_dispspace = TensorFunctionSpace(space, (3, -1))
            # disp = solid_dispspace.function()
            # disp.reshape(-1, 3)[:] = uh.reshape(-1, 3)[is_wall]
            # interface_mesh.nodedata["disp"] = disp.reshape(-1, 3)

            pde.fluid_mesh.to_vtk(f"fluid{i}.vtu")
            # interface_mesh.to_vtk(f"interface{i}.vtu")
            # pde.solid_mesh.to_vtk(f"solid{i}.vtu")
            
            # is_fluid_wall = pde.is_wall_boundary(pde.fluid_mesh.entity('node'))
            # pde.fluid_mesh.node[is_fluid_wall] += disp.reshape(-1, 3)
            # pde.fluid_mesh = TetrahedronMesh(pde.fluid_mesh.node, pde.fluid_mesh.cell)
            # pde.interface_mesh.node += disp.reshape(-1, 3)
            # pde.interface_mesh = TriangleMesh(pde.interface_mesh.node, pde.interface_mesh.cell)
            # pde.solid_mesh.node += uh.reshape(-1, 3)
            # pde.solid_mesh = TetrahedronMesh(pde.solid_mesh.node, pde.solid_mesh.cell)


    def check_convergence(self, last_pressure, last_displacement):
        """
        检查流固耦合的收敛性
        """
        pressure_change = self.fluid_model.calculate_pressure_change(last_pressure)
        displacement_change = self.solid_model.calculate_displacement_change(last_displacement)

        return pressure_change < self.tolerance and displacement_change < self.tolerance