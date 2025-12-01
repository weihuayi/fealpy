from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["StationaryNS2d"]

class StationaryNS2d(CNodeType):
    r"""2D stationary Navier-Stokes equations problem model.

    Inputs:
        example (int): Example number.

    Outputs:
        mu (float): Viscosity coefficient.
        rho (float): Density.
        domain (domain): Computational domain.
        velocity (function): Exact velocity solution.
        pressure (function): Exact pressure solution.
        source (function): Source term.
        velocity_dirichlet (function): Dirichlet boundary condition for velocity.
        pressure_dirichlet (function): Dirichlet boundary condition for pressure.
        is_velocity_boundary (function): Predicate function for velocity boundary regions.
        is_pressure_boundary (function): Predicate function for pressure boundary regions.
    """
    TITLE: str = "二维稳态 NS 方程问题模型"
    PATH: str = "模型.稳态NS"
    DESC: str = """这个节点是一个二维稳态不可压 Navier-Stokes 方程问题模型节点，可根据输入的例子编号生成对应的流体问题，
                包括粘度、密度、求解域、速度与压力的真解、源项以及速度和压力的边界条件与边界判断函数。"""
    INPUT_SLOTS = [
        PortConf("example", DataType.MENU, 0, title="例子编号", default=1, items=[i for i in range(1, 3)])
    ]
    OUTPUT_SLOTS = [
        PortConf("mu", DataType.FLOAT, title="粘度系数"),
        PortConf("rho", DataType.FLOAT, title = "密度"),
        PortConf("domain", DataType.LIST, title="求解域"),
        PortConf("velocity", DataType.FUNCTION, title="速度真解"),
        PortConf("pressure", DataType.FUNCTION, title="压力真解"),
        PortConf("source", DataType.FUNCTION, title="源"),
        PortConf("velocity_dirichlet", DataType.FUNCTION, title="速度边界条件"),
        PortConf("pressure_dirichlet", DataType.FUNCTION, title="压力边界条件"),
        PortConf("is_velocity_boundary", DataType.FUNCTION, title="速度边界"),
        PortConf("is_pressure_boundary", DataType.FUNCTION, title="压力边界")
    ]

    @staticmethod
    def run(example) -> Union[object]:
        from fealpy.cfd.model import CFDTestModelManager

        manager = CFDTestModelManager('stationary_incompressible_navier_stokes')
        model = manager.get_example(example)
        return (model.mu, model.rho, model.domain()) + tuple(
            getattr(model, name)
            for name in ["velocity", "pressure", "source", "velocity_dirichlet", "pressure_dirichlet", "is_velocity_boundary", "is_pressure_boundary"]
        )
