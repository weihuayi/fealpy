
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNS2d", "IncompressibleNS2d"]

class StationaryNSDBC(CNodeType):
    TITLE: str = "稳态 NS 方程边界处理"
    PATH: str = "流体.边界处理"
    INPUT_SLOTS = [
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("velocity_dirichlet", DataType.FUNCTION, title="速度边界条件"),
        PortConf("pressure_dirichlet", DataType.FUNCTION, title="压力边界条件"),
        PortConf("is_velocity_boundary", DataType.FUNCTION, title="速度边界"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界")
    ]
    OUTPUT_SLOTS = [
        PortConf("apply_bc", DataType.FUNCTION, title="边界处理函数")
    ]
    @staticmethod
    def run(uspace, pspace, velocity_dirichlet, pressure_dirichlet, is_velocity_boundary, is_pressure_boundary):
        from fealpy.fem import DirichletBC
        BC = DirichletBC(
            (uspace, pspace), 
            gd=(velocity_dirichlet, pressure_dirichlet), 
            threshold=(is_velocity_boundary, is_pressure_boundary),
            method='interp')
        apply_bc = BC.apply
        return apply_bc
    

class IterativeNSDBC(CNodeType):
    TITLE: str = "非稳态 NS 方程迭代法边界处理"
    PATH: str = "流体.边界处理"
    INPUT_SLOTS = [
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("pspace", DataType.SPACE, title="压力函数空间"),
        PortConf("velocity_dirichlet", DataType.FUNCTION, title="速度边界条件"),
        PortConf("pressure_dirichlet", DataType.FUNCTION, title="压力边界条件"),
        PortConf("is_velocity_boundary", DataType.FUNCTION, title="速度边界"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界")
    ]
    OUTPUT_SLOTS = [
        PortConf("apply_bc", DataType.FUNCTION, title="边界处理函数")
    ]
    @staticmethod
    def run(uspace, pspace, velocity_dirichlet, pressure_dirichlet, is_velocity_boundary, is_pressure_boundary):
        from fealpy.fem import DirichletBC
        from fealpy.decorator import cartesian
        
        def apply_bc(A, b, t=None):
            """
            Apply dirichlet boundary conditions to velocity and pressure.
            """
            if t is None:
                BC = DirichletBC(
                    (uspace, pspace), 
                    gd=(velocity_dirichlet, pressure_dirichlet), 
                    threshold=(is_velocity_boundary, is_pressure_boundary),
                    method='interp')
                A, b = BC.apply(A, b)
            else:
                gd_v = cartesian(lambda p:velocity_dirichlet(p, t))
                gd_p = cartesian(lambda p:pressure_dirichlet(p, t))
                gd = (gd_v, gd_p)
                BC = DirichletBC(
                    (uspace, pspace), 
                    gd=gd, 
                    threshold=(is_velocity_boundary, is_pressure_boundary),
                    method='interp')
                A, b = BC.apply(A, b)
            return A, b
        
        return apply_bc
    
class ProjectDBC(CNodeType):
    TITLE: str = "非稳态 NS 方程投影法边界处理"
    PATH: str = "流体.边界处理"
    INPUT_SLOTS = [
        PortConf("space", DataType.SPACE, title="函数空间"),
        PortConf("dirichlet", DataType.FUNCTION, title="边界条件"),
        PortConf("is_boundary", DataType.FUNCTION, title="边界")
    ]
    OUTPUT_SLOTS = [
        PortConf("apply_bc", DataType.FUNCTION, title="边界处理函数")
    ]
    def run(space, dirichlet, is_boundary):
        from fealpy.fem import DirichletBC
        from fealpy.decorator import cartesian
        def apply_bc(t):
            BC = DirichletBC(space=space,
                    gd = cartesian(lambda p : dirichlet(p, t)),
                    threshold = is_boundary,
                    method = 'interp')
            return BC.apply
        return apply_bc