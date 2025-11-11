from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TetrahedronMesh

try:
    import gmsh
except ImportError:
    raise ImportError("The gmsh package is required for EllipsoidMesher. "
                      "Please install it via 'pip install gmsh'.")


class BoxWithSphereMesher:
    """
    A mesher for generating mesh of a box with spherical cavity.

    Parameters
    box : tuple
        The box defined as (x_min, x_max, y_min, y_max, z_min, z_max).
    sphere_r : int
        The radii of the spherical cavity.
    """
    def __init__(self, box=(-1, 1, -1, 1, -1, 1), sphere_r=0.382):
        if (sphere_r > (box[1] - box[0]) / 2
                or sphere_r > (box[3] - box[2]) / 2
                or sphere_r > (box[5] - box[4]) / 2):
            raise ValueError("sphere_r is too large!")
        self.box = box
        self.center = ((box[1]+box[0])/2, (box[3]+box[2])/2, (box[5]+box[4])/2)
        self.sphere_r = sphere_r

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # 关闭输出
        gmsh.model.add("box_with_sphere")

        # 添加大矩形
        box_gmsh = gmsh.model.occ.addBox(box[0], box[2], box[4], box[1]-box[0], box[3]-box[2], box[5]-box[4])
        # 添加球形空腔
        x0, y0, z0 = self.center
        sphere_gmsh = gmsh.model.occ.addSphere(x0, y0, z0, sphere_r)
        gmsh.model.occ.cut([(3, box_gmsh)], [(3, sphere_gmsh)])
        gmsh.model.occ.synchronize()


    def geo_dimension(self) -> int:
        return 3

    @variantmethod('tri')
    def init_mesh(self, h=0.1):
        box = self.box
        sphere_r = self.sphere_r
        center_point = gmsh.model.occ.addPoint((box[1] + box[0]) / 2, (box[3] + box[2]) / 2, (box[5] + box[4]) / 2)
        gmsh.model.occ.synchronize()
        # 通过距离场设置网格尺寸
        field = gmsh.model.mesh.field
        f_dist = field.add("Distance")
        field.setNumbers(f_dist, "PointsList", [center_point])
        # gmsh.model.mesh.field.setNumber(f_dist, "Sampling", 100)
        p0_field = field.add("Threshold")
        field.setNumber(p0_field, "InField", f_dist)
        field.setNumber(p0_field, "SizeMin",
                        h / (min(box[1] - box[0], box[3] - box[2], box[5] - box[4]) / 2 / sphere_r) ** 2)
        field.setNumber(p0_field, "SizeMax", h)
        field.setNumber(p0_field, "DistMin", sphere_r)
        field.setNumber(p0_field, "DistMax", max((box[1] - box[0])/2,
                                                 (box[3] - box[2])/2, (box[5] - box[4])/2))
        gmsh.model.mesh.field.setAsBackgroundMesh(p0_field)

        gmsh.model.mesh.generate(3)

        # 获取网格
        nodeTags, nodes, _ = gmsh.model.mesh.getNodes()
        nodes = bm.array(nodes, dtype=bm.float64).reshape(-1, 3)
        elementTypes, elementTags, cells = gmsh.model.mesh.getElements(3)
        cells = bm.array(cells[0], dtype=bm.int64).reshape(-1, 4) - 1

        gmsh.finalize()
        return TetrahedronMesh(nodes, cells)
