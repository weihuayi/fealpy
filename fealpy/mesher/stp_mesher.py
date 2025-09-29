from ..backend import bm
from ..decorator import variantmethod

from datetime import datetime
try:
    import gmsh
except ImportError:
    raise ImportError("The gmsh package is required for EllipsoidMesher. "
                      "Please install it via 'pip install gmsh'.")

class STPSurfaceMesher:
    """
    Use Gmsh to generate a surface mesh from an STP file.

    Parameters
    file_path : str
        The path to the STP file.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        gmsh.initialize()
        gmsh.model.add("stp_model")
        # 导入 STP 文件
        gmsh.model.occ.importShapes(file_path)

        # 几何修复（提高鲁棒性，修复拓扑问题）
        gmsh.option.setNumber("Geometry.OCCFixDegenerated", 1)  # 修复退化边缘
        gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 1)   # 修复小边缘
        gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 1)   # 修复小面
        # 同步几何
        gmsh.model.occ.synchronize()
        surfaces = gmsh.model.getEntities(2)
        # 定义物理组（用于保留壳体主要表面）
        if surfaces:
            gmsh.model.addPhysicalGroup(2, [tag for _, tag in surfaces], name="ShellSurface")

    def geo_dimension(self) -> int:
        return 3

    @variantmethod('quad_domain')
    def init_mesh(self, min_size=0.05, max_size=5, output_base="output"):
        # 设置网格密度（自定义 + 自适应）
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)

        # 效率优化：生成四边形网格（Recombine），减少三角形
        gmsh.option.setNumber("Mesh.RecombineAll", 1)  # 重组为四边形
        gmsh.option.setNumber("Mesh.Algorithm", 6)  # 使用 Delaunay 算法，提高效率

        # 生成表面网格（2D，只针对壳体表面）
        gmsh.model.mesh.generate(2)  # 仅生成 2D 网格

        # 优化网格质量
        gmsh.model.mesh.optimize("Netgen")  # 或 "Laplace2D" 对于表面
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        msh_file = f"{output_base}_size{max_size}_{timestamp}.msh"
        gmsh.write(msh_file)
        print(f"表面网格生成完成，已保存到 {msh_file}")

        gmsh.finalize()

    @init_mesh.register('tri')
    def init_mesh(self, min_size=0.05, max_size=5, output_base="output"):
        # 设置网格密度（自定义 + 自适应）
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)

        gmsh.option.setNumber("Mesh.Algorithm", 6)  # 使用 Delaunay 算法，提高效率

        # 生成表面网格（2D，只针对壳体表面）
        gmsh.model.mesh.generate(2)  # 仅生成 2D 网格

        # 优化网格质量
        gmsh.model.mesh.optimize("Netgen")  # 或 "Laplace2D" 对于表面
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        msh_file = f"{output_base}_size{max_size}_{timestamp}.msh"
        gmsh.write(msh_file)
        print(f"表面网格生成完成，已保存到 {msh_file}")

        gmsh.finalize()