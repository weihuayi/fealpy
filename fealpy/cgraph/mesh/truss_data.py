from ..nodetype import CNodeType, PortConf, DataType
from fealpy.backend import backend_manager as bm
from fealpy.mesh import EdgeMesh

__all__ = ["Truss3dData"]

class Truss3dData(CNodeType):
    r"""
    Provides mesh, load, and boundary data for the 25-bar 3D truss problem.

    This node acts as a data source for the truss analysis workflow, encapsulating all
    the necessary information to define a specific benchmark problem.

    Outputs:
        mesh (Mesh): The EdgeMesh object for the 25-bar truss.
        external_load (function): A function that returns the global load vector.
        dirichlet_idx (function): A function that returns the indices of Dirichlet boundary DOFs.
    """
    TITLE: str = "桁架问题数据"
    PATH: str = "网格.构造"
    DESC: str = "提供桁架问题的完整定义，包括网格、载荷函数和边界条件函数。"
    INPUT_SLOTS = []  
    
    OUTPUT_SLOTS = [
        PortConf("mesh", dtype=DataType.MESH, desc="生成的25杆桁架的边网格对象", title="桁架网格"),
        PortConf("external_load", dtype=DataType.FUNCTION, desc="返回全局载荷向量的函数", title="外部载荷函数"),
        PortConf("dirichlet_idx", dtype=DataType.FUNCTION, desc="返回狄利克雷边界条件自由度索引", title="边界索引函数"),
    ]

    @staticmethod
    def run():
        node = bm.array([
            [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540],
            [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
            [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0],
            [-2540, -2540, 0]], dtype=bm.float64)
        edge = bm.array([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4],
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
            [4, 3], [2, 3], [4, 5], [2, 9], [6, 5],
            [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=bm.int32)
        mesh = EdgeMesh(node, edge)

        def external_load(mesh_obj): 
            node_coords = mesh_obj.entity('node')
            GD, NN = 3, mesh_obj.number_of_nodes()
            F = bm.zeros(NN * GD, dtype=bm.float64).reshape(NN, GD)
            idx = bm.where(node_coords[..., 2] == 5080)[0]
            if idx.size > 0:
                F[idx] = bm.array([0.0, 900.0, 0.0], dtype=bm.float64)
            return F.reshape(-1)

        def dirichlet_idx(mesh_obj): 
            return bm.arange(18, 30, dtype=bm.int32)

        return mesh, external_load, dirichlet_idx