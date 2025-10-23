from .nodetype import CNodeType, PortConf, DataType

__all__ = ["VPDecoupling", "UDecoupling"]

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
    DESC: str = """该节点用于将包含速度和压力分量的联合输出向量进行解耦，提取速度场及其各分量与压力场，
                便于后续的流体力学结果分析与可视化处理。"""
    INPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, title="结果"),
        PortConf("uspace", DataType.SPACE, title="速度函数空间"),
        PortConf("mesh", DataType.MESH, title="网格")
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR, title="速度数值解"),
        PortConf("ph", DataType.TENSOR, title="压力数值解"),
        PortConf("uh_x", DataType.TENSOR, title="速度x分量数值解"),
        PortConf("uh_y", DataType.TENSOR, title="速度y分量数值解"),
        PortConf("uh_z", DataType.TENSOR, title="速度z分量数值解")
    ]

    @staticmethod
    def run(out, uspace, mesh):
        from fealpy.backend import backend_manager as bm
        ugdof = uspace.number_of_global_dofs()
        NN = mesh.number_of_nodes()
        uh = out[:ugdof]
        uh = uh.reshape(mesh.GD,-1).T
        uh = uh[:NN,:]
        uh_x = uh[..., 0]
        uh_y = uh[..., 1]
        if mesh.GD == 3:
            uh_z = uh[..., 2]
        else:
            uh_z = bm.zeros_like(uh_x)
        ph = out[ugdof:]

        return uh, ph, uh_x, uh_y, uh_z


class UDecoupling(CNodeType):
    r"""Decouple translational and rotational displacement components 
    from the combined output vector.
    
    Inputs:
        out (tensor): Combined displacement vector of all nodes.Each node contains six components in the order 
            [u, v, w, θx, θy, θz].
            
    Outputs:
        uh (tensor): Translational displacement field (X, Y, Z components).
        theta_xyz (tensor): Rotational displacement field (rotations around X, Y, Z axes).
    """
    TITLE: str = "位移解耦"
    PATH: str = "后处理.解耦"
    DESC: str = "将平动位移和转动位移做解耦处理"
    INPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, 1, desc="六个自由度的位移", title="结果")
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR, desc="X,Y,Z三个方向上的位移", title="平动位移"),
        PortConf("theta_xyz", DataType.TENSOR, desc="X,Y,Z三个方向的弯曲和剪切产生的位移", title="转动位移"),
    ]

    @staticmethod
    def run(out):
        u = out.reshape(-1, 6)
        
        uh = u[:, :3]
        theta_xyz = u[:, 3:]

        return uh, theta_xyz
