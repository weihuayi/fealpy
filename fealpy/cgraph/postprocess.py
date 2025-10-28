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
        
        uh = u[:, 3:]
        theta_xyz = u[:, :3]

        return uh, theta_xyz
    
class BeamPostprocess(CNodeType):
    r"""
    Postprocess the beam element results to extract nodal displacements and rotations.

    Inputs:
    out (tensor): Combined displacement vector of all nodes. Each node contains six components in the order
    [u, θx, v, θy, w, θz].
    Outputs:
    uh (tensor): Nodal translational displacements (u, v, w).
    theta_xyz (tensor): Nodal rotational displacements (θx, θy, θz).
    """
    TITLE: str = "梁单元后处理"
    PATH: str = "后处理.梁单元"
    DESC: str = "提取梁单元的节点位移和旋转"
    INPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, 1, desc="六个自由度的位移", title="结果")
    ]
    OUTPUT_SLOTS = [   
        PortConf("uh", DataType.TENSOR, desc="节点平动位移", title="平动位移"),
        PortConf("theta_xyz", DataType.TENSOR, desc="节点转动位移", title="转动位移"),
    ]
    @staticmethod
    def run(out):   
        uh = out[::2]
        theta = out[1::2]
        return uh, theta

class TrussPostprocess(CNodeType):
    r"""Calculates the displacement of each node and the strain and stress of each rod element 
    based on the raw displacement vector output by the solver and material parameters.

    Inputs:
        uh (tensor): Raw displacement vector output by the solver.
        mesh (mesh): Mesh containing node and cell information.
        E (float): Elastic modulus of the rod.
    Outputs:
        strain (tensor): Strain of each rod element.
        stress (tensor): Stress of each rod element.
        uh_reshaped (tensor): Reshaped displacement tensor (NN, GD).
    """
    TITLE: str = "桁架后处理"
    PATH: str = "后处理.位移应力应变"
    DESC: str = "根据求解器输出的原始位移向量和材料参数，计算每个节点的位移和每个杆单元的应变应力"
    INPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR, 1, desc="求解器输出的原始位移向量", title="位移向量"),
        PortConf("mesh", DataType.MESH, 1, desc="包含节点和单元信息的网格", title="网格"),
        PortConf("E", DataType.FLOAT, 1, desc="杆的弹性模量", title="弹性模量"),
    ]
    OUTPUT_SLOTS = [
        PortConf("strain", DataType.TENSOR, desc="每个杆单元的应变", title="应变"),
        PortConf("stress", DataType.TENSOR, desc="每个杆单元的应力", title="应力"),
        PortConf("uh_reshaped", DataType.TENSOR, desc="重塑后的位移张量 (NN, GD)", title="重塑位移")
    ]

    @staticmethod
    def run(uh, mesh, E):
        from fealpy.backend import backend_manager as bm
        
        uh_reshaped = uh.reshape(-1, 3) 

        edge = mesh.entity('edge')
        l = mesh.edge_length()
        tan = mesh.edge_tangent()
        unit_tan = tan / l.reshape(-1, 1)

        u_edge = uh_reshaped[edge]
        delta_u = u_edge[:, 1, :] - u_edge[:, 0, :]
        delta_l = bm.einsum('ij,ij->i', delta_u, unit_tan)
        strain = delta_l / l
        stress = E * strain
        
        return strain, stress, uh_reshaped
