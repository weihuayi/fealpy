from typing import Sequence
from ..backend import backend_manager as bm
from ..decorator import variantmethod
from ..mesh import TriangleMesh,QuadrangleMesh

class TorusMesher:
    def __init__(self):
        pass

    def geo_dimension(self) -> int:
        return 2
    
    @variantmethod('tri')
    def init_mesh(self, R: float = 1.0, r: float = 0.3,
                  x: float = 0.0, y: float = 0.0, z: float = 0.0,
                  h: float = 0.1, 
                  ax: float = 0.0, ay: float = 0.0, az: float = 1.0):
        """
        Using Gmsh to generate a 3D surface triangular mesh for a torus (ring surface).
        Parameters:
            R(float): major radius (distance from torus center to tube center)
            r(float): minor radius (tube radius)
            center(list): center of the torus [cx, cy, cz]
            h(float): mesh size
            axis(list): axis direction of the torus (e.g. [0,0,1] for z-axis)
            show_gui(bool): whether to show Gmsh GUI
        Returns:
            mesh(TriangleMesh): triangle mesh of the torus
        """
        import gmsh
        gmsh.initialize()
        gmsh.model.add("Torus")
        cx, cy, cz = x, y, z

        # 添加一个圆环面体
        tag = gmsh.model.occ.addTorus(cx, cy, cz, R, r , tag=1)

        axis = bm.from_numpy([ax, ay, az])
        axis = axis / bm.linalg.norm(axis)
        
        z_axis = bm.array([0, 0, 1], dtype=float)
        if not bm.allclose(axis, z_axis):
            # 旋转轴为 z_axis x axis
            rot_axis = bm.cross(z_axis, axis)
            rot_angle = bm.arccos(bm.clip(bm.dot(z_axis, axis), -1.0, 1.0))
            rot_axis = rot_axis / bm.linalg.norm(rot_axis)
            gmsh.model.occ.rotate([(3, tag)], cx, cy, cz, rot_axis[0], rot_axis[1], rot_axis[2], rot_angle)
        gmsh.model.occ.synchronize()

        # 网格尺寸设置
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        # 生成三维网格
        gmsh.model.mesh.generate(2)

        # 提取网格信息
        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = bm.from_numpy(node_tags)
        node_coords = bm.from_numpy(node_coords)
        node = node_coords.reshape((-1, 3))

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
    def init_mesh(self, R: float = 1.0, r: float = 0.3, 
                  x: float = 0.0, y: float = 0.0, z: float = 0.0,
                  h: float = 0.1, 
                  ax: float = 0.0, ay: float = 0.0, az: float = 1.0):
        """
        Using Gmsh to generate a 3D surface triangular mesh for a torus (ring surface).
        Parameters:
            R(float): major radius (distance from torus center to tube center)
            r(float): minor radius (tube radius)
            center(list): center of the torus [cx, cy, cz]
            h(float): mesh size
            axis(list): axis direction of the torus (e.g. [0,0,1] for z-axis)
            show_gui(bool): whether to show Gmsh GUI
        Returns:
            mesh(QuadrangleMesh): quadrangle mesh of the torus
        """
        import gmsh
        gmsh.initialize()
        gmsh.model.add("Torus")
        cx, cy, cz = x, y, z

        # 添加一个圆环面体
        tag = gmsh.model.occ.addTorus(cx, cy, cz, R, r , tag=1)
        
        axis = bm.from_numpy([ax, ay, az])
        axis = axis / bm.linalg.norm(axis)
        
        z_axis = bm.array([0, 0, 1], dtype=float)
        if not bm.allclose(axis, z_axis):
            # 旋转轴为 z_axis x axis
            rot_axis = bm.cross(z_axis, axis)
            rot_angle = bm.arccos(bm.clip(bm.dot(z_axis, axis), -1.0, 1.0))
            rot_axis = rot_axis / bm.linalg.norm(rot_axis)
            gmsh.model.occ.rotate([(3, tag)], cx, cy, cz, rot_axis[0], rot_axis[1], rot_axis[2], rot_angle)
        gmsh.model.occ.synchronize()
        # 获取所有面
        surfaces = gmsh.model.getEntities(2)
        surf_tag = surfaces[0][1]
        # 网格尺寸设置
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
        gmsh.model.mesh.setRecombine(2, surf_tag)
        # 生成三维网格
        gmsh.model.mesh.generate(2)

        # 提取网格信息
        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = bm.from_numpy(node_tags)
        node_coords = bm.from_numpy(node_coords)
        node = node_coords.reshape((-1, 3))

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