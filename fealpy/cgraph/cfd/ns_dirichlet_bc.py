
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNS2d", "IncompressibleNS2d"]

class StationaryNSDBC(CNodeType):
    r"""Boundary condition handler for the stationary Navier-Stokes equations.

    This computational node constructs and applies Dirichlet boundary conditions 
    for both velocity and pressure fields in the stationary Navier-Stokes equations.
    It provides a unified interface to generate the boundary condition application function 
    used by subsequent solver nodes.

    Inputs:
        uspace(space): Function space for the velocity field.
        pspace(space): Function space for the pressure field.
        velocity_dirichlet (function): Dirichlet boundary function for velocity.
        pressure_dirichlet (function): Dirichlet boundary function for pressure.
        is_velocity_boundary (function): Predicate function identifying velocity boundary regions.
        is_pressure_boundary (function): Predicate function identifying pressure boundary regions.

    Outputs:
        apply_bc (function): Function that applies the specified Dirichlet boundary conditions 
            to the assembled system matrices and right-hand side vectors.
    """
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
    

class IterativeDBC(CNodeType):
    r"""Boundary condition handler for unsteady Navier-Stokes iterative solvers.

    This computational node defines a unified boundary condition processor 
    for unsteady (time-dependent) incompressible Navier-Stokes equations
    when iterative schemes (e.g. Oseen, or Newton methods) are used.  
    It provides a callable function that applies both velocity and pressure 
    Dirichlet boundary conditions at a given time step.

    Inputs:
        uspace(space): Function space for the velocity field.
        pspace(space): Function space for the pressure field.
        velocity_dirichlet (function): Dirichlet boundary condition for velocity.
        pressure_dirichlet (function): Dirichlet boundary condition for pressure.
        is_velocity_boundary (function): Predicate function identifying velocity boundary regions.
        is_pressure_boundary (function): Predicate function identifying pressure boundary regions.

    Outputs:
        apply_bc (function): Function to apply the time-dependent Dirichlet boundary 
            conditions to system matrices and right-hand side vectors.
    """
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
    r"""Boundary condition handler for unsteady Navier-Stokes projection methods. 

    Inputs:
        space(space): Function space for the velocity field.
        dirichlet (function): Dirichlet boundary condition for velocity.
        is_boundary (function): Predicate function identifying velocity boundary regions.
    Outputs:
        apply_bc (function): Function to apply the time-dependent Dirichlet boundary 
            conditions to system matrices and right-hand side vectors.
    """
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