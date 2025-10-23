from .nodetype import CNodeType, PortConf, DataType

__all__ = ["VPDecoupling"]

class VPDecoupling(CNodeType):
    r"""Decouple velocity and pressure components from the combined output vector.

    Inputs:
        out (tensor): Combined output vector containing velocity and pressure components.
        uspace (space): Function space for the velocity field.
        mesh (mesh): Computational mesh.
    Outputs:
        uh (tensor): Numerical velocity field.
        u_x (tensor): x-component of the velocity field.
        u_y (tensor): y-component of the velocity field.
        ph (tensor): Numerical pressure field.
    """
    TITLE: str = "速度-压力解耦"
    PATH: str = "后处理.解耦"
    INPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="结果"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR, title="速度数值解"),
        PortConf("u_x", DataType.TENSOR, title="速度x分量数值解"),
        PortConf("u_y", DataType.TENSOR, title="速度y分量数值解"),
        PortConf("ph", DataType.TENSOR, title="压力数值解")
    ]

    @staticmethod
    def run(out, uspace, mesh):
        ugdof = uspace.number_of_global_dofs()
        NN = mesh.number_of_nodes()
        uh = out[:ugdof]
        uh = uh.reshape(mesh.GD,-1).T
        uh = uh[:NN,:]
        u_x = out[:int(ugdof/2)]
        u_x = u_x[:NN]
        u_y = out[int(ugdof/2):ugdof]
        u_y = u_y[:NN]
        ph = out[ugdof:]

        return uh, u_x, u_y, ph
