from typing import Sequence
from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TriangleMesh,QuadrangleMesh

class EllipseMesher:
    def __init__(self):
        pass

    def geo_dimension(self) -> int:
        return 2
    
    @variantmethod('tri')
    def init_mesh(self, a: float, b: float, x: float = 0.0, y: float = 0.0, h: float = 0.1, theta: float = 0.0):
        """
        Using Gmsh to generate a 2D triangular mesh for an ellipse.
        Parameters:
            a(float): semi-major axis
            b(float): semi-minor axis
            x(float): center x-coordinate
            y(float): center y-coordinate
            theta(float): rotation angle in radians
            h(float): mesh size
        Returns:
            mesh(TriangleMesh): triangle mesh of the ellipse
        """
        import gmsh
        gmsh.initialize()
        gmsh.model.add('Ellipse')
        cx = x
        cy = y
        rotate_matrix = bm.array([[bm.cos(theta), -bm.sin(theta)],
                                [bm.sin(theta),  bm.cos(theta)],])
        # 四个点坐标
        axis_points = bm.array([[a, 0],
                                [0, b],
                                [-a, 0],
                                [0, -b],])
        
        # 旋转平移椭圆的4个关键点
        rotated_points = (rotate_matrix @ axis_points.T).T + bm.array([cx, cy])
        # 定义椭圆的4个关键点
        p1 = gmsh.model.geo.addPoint( rotated_points[0, 0], rotated_points[0, 1], 0, h ,tag = 1)
        p2 = gmsh.model.geo.addPoint( rotated_points[1, 0], rotated_points[1, 1], 0, h ,tag = 2)
        p3 = gmsh.model.geo.addPoint( rotated_points[2, 0], rotated_points[2, 1], 0, h ,tag = 3)
        p4 = gmsh.model.geo.addPoint( rotated_points[3, 0], rotated_points[3, 1], 0, h ,tag = 4)
        c = gmsh.model.geo.addPoint( cx, cy, 0, h ,tag = 5) # 椭圆中心

        # 添加4段椭圆弧
        c1 = gmsh.model.geo.addEllipseArc(p1, c, p2, p2 ,tag = 6)
        c2 = gmsh.model.geo.addEllipseArc(p2, c, p3, p3 ,tag = 7)
        c3 = gmsh.model.geo.addEllipseArc(p3, c, p4, p4 ,tag = 8)
        c4 = gmsh.model.geo.addEllipseArc(p4, c, p1, p1 ,tag = 9)
        # 组成曲线环和面
        cl = gmsh.model.geo.addCurveLoop([c1,c2,c3,c4])
        surf = gmsh.model.geo.addPlaneSurface([cl])

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        # 提取网格信息
        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = bm.from_numpy(node_tags)
        node_coords = bm.from_numpy(node_coords)
        node = node_coords.reshape((-1, 3))[:, :2]

        # 节点编号映射
        nodetags_map = dict({int(j): i for i, j in enumerate(node_tags)})

        # 获取单元信息
        cell_type = 2  # 三角形单元的类型编号为 2
        cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)

        # 节点编号映射到单元
        evid = bm.array([nodetags_map[int(j)] for j in cell_connectivity])
        cell = evid.reshape((cell_tags.shape[-1], -1))

        gmsh.finalize()
        mesh = TriangleMesh(node, cell)
        return mesh
    
    @init_mesh.register('quad')
    def init_mesh(self, a: float, b: float, x: float = 0.0, y: float = 0.0, h: float = 0.1, theta: float = 0.0):
        """
        Using Gmsh to generate a 2D triangular mesh for an ellipse.
        Parameters:
            a(float): semi-major axis
            b(float): semi-minor axis
            x(float): center x-coordinate
            y(float): center y-coordinate
            theta(float): rotation angle in radians
            h(float): mesh size
        Returns:
            mesh(QuadrangleMesh): quadrangle mesh of the ellipse
        """
        import gmsh
        gmsh.initialize()
        gmsh.model.add('Ellipse')
        cx = x
        cy = y
        rotate_matrix = bm.array([[bm.cos(theta), -bm.sin(theta)],
                                [bm.sin(theta),  bm.cos(theta)],])
        # 四个点坐标
        axis_points = bm.array([[a, 0],
                                [0, b],
                                [-a, 0],
                                [0, -b],])
        
        # 旋转平移椭圆的4个关键点
        rotated_points = (rotate_matrix @ axis_points.T).T + bm.array([cx, cy])
        # 定义椭圆的4个关键点
        p1 = gmsh.model.geo.addPoint( rotated_points[0, 0], rotated_points[0, 1], 0, h ,tag = 1)
        p2 = gmsh.model.geo.addPoint( rotated_points[1, 0], rotated_points[1, 1], 0, h ,tag = 2)
        p3 = gmsh.model.geo.addPoint( rotated_points[2, 0], rotated_points[2, 1], 0, h ,tag = 3)
        p4 = gmsh.model.geo.addPoint( rotated_points[3, 0], rotated_points[3, 1], 0, h ,tag = 4)
        c = gmsh.model.geo.addPoint( cx, cy, 0, h ,tag = 5) # 椭圆中心

        # 添加4段椭圆弧
        c1 = gmsh.model.geo.addEllipseArc(p1, c, p2, p2 ,tag = 6)
        c2 = gmsh.model.geo.addEllipseArc(p2, c, p3, p3 ,tag = 7)
        c3 = gmsh.model.geo.addEllipseArc(p3, c, p4, p4 ,tag = 8)
        c4 = gmsh.model.geo.addEllipseArc(p4, c, p1, p1 ,tag = 9)
        # 组成曲线环和面
        cl = gmsh.model.geo.addCurveLoop([c1,c2,c3,c4])
        surf = gmsh.model.geo.addPlaneSurface([cl])

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.setRecombine(2, surf)
        gmsh.model.mesh.generate(2)

        # 提取网格信息
        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = bm.from_numpy(node_tags)
        node_coords = bm.from_numpy(node_coords)
        node = node_coords.reshape((-1, 3))[:, :2]

        # 节点编号映射
        nodetags_map = dict({int(j): i for i, j in enumerate(node_tags)})

        # 获取单元信息
        cell_type = 3  # 四边形单元的类型编号为 3
        cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)

        # 节点编号映射到单元
        evid = bm.array([nodetags_map[int(j)] for j in cell_connectivity])
        cell = evid.reshape((cell_tags.shape[-1], -1))
        gmsh.finalize()
        mesh = QuadrangleMesh(node, cell)
        return mesh