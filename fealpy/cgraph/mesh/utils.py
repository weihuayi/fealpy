from typing import Type
import importlib


def get_mesh_class(mesh_type: str) -> Type:
    m = importlib.import_module(f"fealpy.mesh.{mesh_type}_mesh")
    mesh_class_name = mesh_type[0].upper() + mesh_type[1:] + "Mesh"
    return getattr(m, mesh_class_name)


from ..nodetype import CNodeType, PortConf, DataType

__all__ = ["MatMatrixReader"]
class MatMatrixReader(CNodeType):
    r"""Create a matrix from *.mat file.

    This node reads a *.mat file and creates a matrix from it. The matrix can be used for various applications such as
    solving linear equations, solving partial differential equations, or performing other numerical operations.

    Inputs:
        input_mat_file (str): Path to the *.mat file.

    Outputs:
        output_matrix (MatrixType): The matrix object created.
    """
    TITLE: str = "MAT 矩阵读取"
    PATH: str = "矩阵.构造"
    INPUT_SLOTS = [
        PortConf("input_mat_file", DataType.STRING, title="文件路径", default="")
    ]
    OUTPUT_SLOTS = [
        PortConf("output_matrix", DataType.MENU, title="矩阵")
    ]

    @staticmethod
    def run(input_mat_file):
        
        from scipy.io import loadmat
        matrix = loadmat(input_mat_file)
        
        return matrix 
     