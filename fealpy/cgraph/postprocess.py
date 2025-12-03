from .nodetype import CNodeType, PortConf, DataType

__all__ = ["VPDecoupling", "UDecoupling", "AntennaPostprocess","GearboxPostprocess"]

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
    PATH: str = "postprocess"
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
    ]

    @staticmethod
    def run(out, uspace, mesh):
        from fealpy.backend import backend_manager as bm
        ugdof = uspace.number_of_global_dofs()
        NN = mesh.number_of_nodes()
        uh = out[:ugdof]
        uh = uh.reshape(mesh.GD,-1).T
        ph = out[ugdof:]
        return uh, ph


class UDecoupling(CNodeType):
    r"""Decouple translational and rotational displacement components 
    from the combined output vector.
    
    Inputs:
        out (tensor): Combined displacement vector of all nodes. 
        node_ldof (INT): Number of local degrees of freedom (DOFs) per node. 
        type (MENU): Type of finite element.
            
    Outputs:
        uh (tensor): Translational displacement field.
        theta (tensor): Rotational displacement gfield.
    """
    TITLE: str = "位移后处理"
    PATH: str = "postprocess"
    DESC: str = "将模型的平动位移和转动位移做后处理"
    INPUT_SLOTS = [
        PortConf("out", DataType.TENSOR, 1, desc="求解器输出的原始位移向量", title="位移向量"),
        PortConf("node_ldof", DataType.MENU, 0, desc="节点的自由度个数", title="自由度长度", default=2, items=[2, 3, 4, 6]),
        PortConf("type", DataType.MENU, 0, desc="单元的类型", title="单元类型", default="Truss", items=["Truss", "Euler_beam", "Timo_beam"]),
    ]
    OUTPUT_SLOTS = [
        PortConf("uh", DataType.TENSOR,  title="平动位移"),
        PortConf("theta", DataType.TENSOR, title="转动位移"),
    ]

    @staticmethod
    def run(out, node_ldof, type):
        
        u = out.reshape(-1, node_ldof)
        
        if type == "Truss":
            uh = u
            theta = None
        elif type == "Euler_beam":
            uh = u[:, :1]
            theta = u[:, 1:]
        elif type == "Timo_beam":
            uh = u[:, :3]
            theta = u[:, 3:]
        else: 
            raise ValueError(f'post-processing for this type of displacement is not supported yet.')

        return uh, theta


class AntennaPostprocess(CNodeType):
    r"""Compute the real-valued electric field at each antenna element 
    from the complex-valued finite element solution.
    
    Inputs:
        uh (tensor): Complex-valued finite element solution vector (nodal degrees of freedom).
        space (Space): Finite element space containing mesh and basis function information.
            
    Outputs:
        E (tensor): Real part of the electric field evaluated at the centroid of each element.
    """

    TITLE: str = "天线单元后处理"
    PATH: str = "postprocess"
    DESC: str = "将天线的自由度平分到各自单元"
    INPUT_SLOTS = [
        PortConf("uh", DataType.FUNCTION, 1, desc="求解器输出的复数场自由度", title="有限元解"),
        PortConf("mesh", DataType.MESH, 1, title="网格"),
    ]
    OUTPUT_SLOTS = [
        PortConf("E", DataType.TENSOR, desc="各单元重心处的电场强度（实部）", title="单元电场"),
    ]

    @staticmethod
    def run(uh, mesh):
        from fealpy.backend import backend_manager as bm
        from fealpy.functionspace import Function
        from fealpy.functionspace import FirstNedelecFESpace
        space = FirstNedelecFESpace(mesh = mesh, p=1)
        bc = bm.array([[1/3, 1/3, 1/3, 1/3]], dtype=bm.float64)
        gdof = space.number_of_global_dofs()

        if isinstance(uh, Function):
            uh_real = uh.copy()
            uh_real[:] = bm.real(uh[:gdof])
        else:
            uh_real = space.function()
            uh_real[:] = bm.real(uh[:gdof])

        val = space.value(uh_real, bc)
        E = val.reshape(mesh.number_of_cells(), -1)

        return E


class GearboxPostprocess(CNodeType):
    TITLE: str = "变速箱后处理"
    PATH: str = "postprocess"
    DESC: str = "将模态特征向量映射到网格节点并计算固有频率"

    INPUT_SLOTS = [
        PortConf("mesh", DataType.SPACE, 1, desc="有限元网格", title="网格"),
        PortConf("vals", DataType.TENSOR, 1, desc="特征值", title="特征值"),
        PortConf("vecs", DataType.TENSOR, 1, desc="特征向量", title="特征向量"),
        PortConf("NS", DataType.TENSOR, 1, desc="自由度划分信息", title="自由度划分"),
        PortConf("G", DataType.TENSOR, 1, desc="耦合矩阵", title="耦合矩阵"),
        PortConf("output_file", DataType.STRING, 0, desc="输出文件路径", title="输出文件路径", default="/home"),
    ]

    OUTPUT_SLOTS = [
        PortConf("freqs", DataType.TENSOR, desc="固有频率 (Hz)", title="固有频率"),
        PortConf("eigvecs", DataType.TENSOR, desc="映射后的特征向量", title="特征向量"),
        PortConf("output_file", DataType.STRING, desc="输出文件路径", title="输出文件路径"),
    ]

    @staticmethod
    def run(mesh, vals, vecs, NS, G, output_file):
        from ..backend import backend_manager as bm

        freqs = bm.sqrt(vals) / (2 * bm.pi)

        NN = mesh.number_of_nodes()
        isFreeNode = mesh.data.get_node_data('isFreeNode')
        isFreeDof = bm.repeat(isFreeNode, 3)
        isCSNode = mesh.data.get_node_data('isCSNode')
        isCSDof = bm.repeat(isCSNode, 3)

        mapped_eigvecs = []
        start = sum(NS[0:-2])
        end = start + NS[-2]
        from pathlib import Path
        export_dir = Path(output_file).expanduser().resolve()
        export_dir.mkdir(parents=True, exist_ok=True)
        fname = export_dir / f"test_{str(0).zfill(10)}.vtu"
        for i, val in enumerate(vecs):
            phi = bm.zeros((NN * 3,), dtype=bm.float64)

            idx, = bm.where(isFreeDof)
            if (end - start) == idx.shape[0]:
                phi = bm.set_at(phi, idx, val[start:end])

            idx, = bm.where(isCSDof)
            phi = bm.set_at(phi, idx, G @ val[end:])
            phi = phi.reshape((NN, 3))
            mapped_eigvecs.append(phi)
     
            mesh.nodedata[f'eigenvalue-{i}-{vals[i]:0.5e}'] = phi

        eigvecs = bm.array(mapped_eigvecs)
        mesh.to_vtk(fname=fname)
        output_file = str(fname)
        return freqs, eigvecs, output_file
