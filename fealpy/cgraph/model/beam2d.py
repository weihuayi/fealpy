from typing import Union
from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["Beam2d"]


class Beam2d(CNodeType):
    r"""2D Euler-Bernoulli Beam-Axle Geometry Model.

     Inputs:
        None

    Outputs:
        f: External load vector (array or callable).
        dirichlet: Dirichlet boundary condition function.
        dirichlet_dof_index: Indices of DOFs with Dirichlet boundary conditions (array or callable).
    """
    TITLE: str = "欧拉梁几何参数模型"
    PATH: str = "模型.几何参数"
    DESC: str = "定义欧拉梁的几何结构及边界条件函数"
    INPUT_SLOTS = []
    OUTPUT_SLOTS = [
        PortConf("f", DataType.INT, desc="外部载荷向量", title="外部载荷"),
        PortConf("dirichlet", DataType.FUNCTION, desc="Dirichlet 边界条件的函数", title="边界条件"),
        PortConf("dirichlet_dof_index", DataType.FUNCTION, desc="Dirichlet 自由度索引的函数", title="边界自由度索引")
        
    ]

    @staticmethod
    def run():
        from fealpy.csm.model.beam.euler_bernoulli_beam_data_2d import EulerBernoulliBeamData2D
        model = EulerBernoulliBeamData2D()
        return (model.f,)+tuple(
            getattr(model, name)
            for name in ["dirichlet", "dirichlet_dof_index"]
        )
       