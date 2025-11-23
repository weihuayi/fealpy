from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TetrahedronMesh, QuadrangleMesh, TriangleMesh

try:
    import gmsh
except ImportError:
    raise ImportError("The gmsh package is required for EllipsoidMesher. "
                      "Please install it via 'pip install gmsh'.")



class EllipsoidMesher:
    """
    A mesher for generating meshes of an ellipsoid.

    Parameters
    center : tuple
        The center of the ellipsoid (x, y, z).
    radii : tuple
        The radii of the ellipsoid along the x, y, and z axes (rx, ry, rz).
    """
    def __init__(self, center=(0, 0, 0), radii=(2, 1, 0.5)):
        self.center = center
        self.radii = radii

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # 关闭输出
        gmsh.model.add("ellipsoid_surface")

        # 添加椭球（使用缩放的球）
        x0, y0, z0 = center
        rx, ry, rz = radii
        sph = gmsh.model.occ.addSphere(x0, y0, z0, 1.0)
        gmsh.model.occ.dilate([(3, sph)], x0, y0, z0, rx, ry, rz)
        gmsh.model.occ.synchronize()


    def geo_dimension(self) -> int:
        return 3

    @variantmethod('surface_tri')
    def init_mesh(self, lc=0.1):
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(2)

        # 获取网格
        nodeTags, nodes, _ = gmsh.model.mesh.getNodes()
        nodes = bm.array(nodes, dtype=bm.float64).reshape(-1, 3)
        elementTypes, elementTags, cells = gmsh.model.mesh.getElements(2)
        cells = bm.array(cells[0], dtype=bm.int64).reshape(-1, 3) - 1

        gmsh.finalize()
        return TriangleMesh(nodes, cells)

    @init_mesh.register('surface_quad')
    def init_mesh(self, lc=0.1):
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.model.mesh.generate(2)

        # 获取网格
        nodeTags, nodes, _ = gmsh.model.mesh.getNodes()
        nodes = bm.array(nodes, dtype=bm.float64).reshape(-1, 3)
        elementTypes, elementTags, cells = gmsh.model.mesh.getElements(2)
        cells = bm.array(cells[0], dtype=bm.int64).reshape(-1, 4) - 1

        gmsh.finalize()
        return QuadrangleMesh(nodes, cells)

    @init_mesh.register('volume')
    def init_mesh(self, lc=0.1):
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
        gmsh.model.mesh.generate(3)

        # 获取网格
        nodeTags, nodes, _ = gmsh.model.mesh.getNodes()
        nodes = bm.array(nodes, dtype=bm.float64).reshape(-1, 3)
        elementTypes, elementTags, cells = gmsh.model.mesh.getElements(3)
        cells = bm.array(cells[0], dtype=bm.int64).reshape(-1, 4) - 1

        gmsh.finalize()
        return TetrahedronMesh(nodes, cells)
