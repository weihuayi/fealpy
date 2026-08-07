import gmsh
import math
import numpy as np

class PipeGeometry:
    def __init__(self):
        # 几何数据参数化
        self.R = 0.5        # 管道半径 0.5m
        self.D = 1.0        # 管道直径 1.0m
        self.Rc = 2.8       # 曲率半径 2.8 * D = 2.8m
        self.L_up = 10.0    # 上游直管段长度 10m
        self.L_down = 15.0  # 下游直管段长度 15m
        
        self.volume_tag = None
        self.wall_tags = [] # 保存壁面的 tags 供边界层使用

    def build(self):
        """构建 90 度弯管几何模型 (中心线扫掠法)"""
        gmsh.initialize()
        gmsh.model.add("Benchmark_90_Degree_Bend_Native_X")

        # ==========================================
        # 1. 构建中心线轨迹 (Wire)
        # ==========================================
        p1 = gmsh.model.occ.addPoint(-self.L_up, 0, 0)
        p2 = gmsh.model.occ.addPoint(0, 0, 0)
        p_center = gmsh.model.occ.addPoint(0, self.Rc, 0)
        p3 = gmsh.model.occ.addPoint(self.Rc, self.Rc, 0)
        p4 = gmsh.model.occ.addPoint(self.Rc, self.Rc + self.L_down, 0)

        l1 = gmsh.model.occ.addLine(p1, p2)                     
        arc = gmsh.model.occ.addCircleArc(p2, p_center, p3)     
        l2 = gmsh.model.occ.addLine(p3, p4)                     

        wire = gmsh.model.occ.addWire([l1, arc, l2])

        # ==========================================
        # 2. 构建截面并扫掠成体 (Pipe)
        # ==========================================
        disk = gmsh.model.occ.addDisk(-self.L_up, 0, 0, self.R, self.R, zAxis=[1, 0, 0])
        pipe = gmsh.model.occ.addPipe([(2, disk)], wire)
        
        gmsh.model.occ.synchronize()
        self.volume_tag = pipe[0][1]
        self.classify_boundaries()

    def classify_boundaries(self):
        """判断并划分 入口、出口 和 壁面边界"""
        surfaces = gmsh.model.getBoundary([(3, self.volume_tag)], oriented=False)
        
        inlet_tags = []
        outlet_tags = []
        wall_tags = []

        for dim, tag in surfaces:
            com = gmsh.model.occ.getCenterOfMass(dim, tag)
            
            if abs(com[0] - (-self.L_up)) < 1e-3:
                inlet_tags.append(tag)
            elif abs(com[1] - (self.Rc + self.L_down)) < 1e-3:
                outlet_tags.append(tag)
            else:
                wall_tags.append(tag)

        gmsh.model.addPhysicalGroup(2, inlet_tags, name="Inlet")
        gmsh.model.addPhysicalGroup(2, outlet_tags, name="Outlet")
        gmsh.model.addPhysicalGroup(2, wall_tags, name="Wall")
        gmsh.model.addPhysicalGroup(3, [self.volume_tag], name="FluidDomain")
        
        # 将 wall_tags 存入实例变量
        self.wall_tags = wall_tags
        print("几何构建与边界划分完成！")


class PipeMesh:
    def __init__(self, geometry: PipeGeometry, mesh_size=0.4, 
                 bl_enable=True, bl_size=0.02, bl_thickness=0.15):
        self.geom = geometry
        self.mesh_size = mesh_size
        
        # 边界层参数接口
        self.bl_enable = bl_enable
        self.bl_size = bl_size           # 边界层首层/最小网格高度
        self.bl_thickness = bl_thickness # 过渡到内部最大网格的边界距离

    def generate_mesh(self):
        if self.geom.volume_tag is None:
            raise ValueError("几何未构建！请先调用 geometry.build()")

        print(f"开始生成 3D 网格，全局最大尺寸设定为: {self.mesh_size} ...")
        
        # ==========================================
        # 使用 Distance + Threshold 配置纯四面体边界层
        # ==========================================
        if self.bl_enable and hasattr(self.geom, 'wall_tags') and len(self.geom.wall_tags) > 0:
            print(f"应用四面体贴体边界层: 最细尺寸 {self.bl_size}, 过渡距离 {self.bl_thickness} ...")
            
            # 1. 距离场：计算每个点到壁面 (wall_tags) 的最短距离
            dist_id = 1
            gmsh.model.mesh.field.add("Distance", dist_id)
            gmsh.model.mesh.field.setNumbers(dist_id, "SurfacesList", self.geom.wall_tags)
            
            # 2. 阈值场：根据距离场的值映射网格尺寸
            thresh_id = 2
            gmsh.model.mesh.field.add("Threshold", thresh_id)
            gmsh.model.mesh.field.setNumber(thresh_id, "IField", dist_id)
            
            # 距壁面近处的最小网格尺寸 (bl_size)
            gmsh.model.mesh.field.setNumber(thresh_id, "LcMin", self.bl_size)
            # 距壁面远处的全局最大网格尺寸 (mesh_size)
            gmsh.model.mesh.field.setNumber(thresh_id, "LcMax", self.mesh_size)
            
            # 距离壁面多远时开始逐步增大尺寸
            gmsh.model.mesh.field.setNumber(thresh_id, "DistMin", self.bl_size)
            # 距离壁面多远时网格尺寸达到 LcMax 并停止增长
            gmsh.model.mesh.field.setNumber(thresh_id, "DistMax", self.bl_thickness)
            
            # 3. 将其设置为全局背景网格尺寸场
            gmsh.model.mesh.field.setAsBackgroundMesh(thresh_id)
            
            # 禁用默认的曲率和延伸散布逻辑，强制采用我们的阈值场
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.Algorithm3D", 10) # 采用 HXT 优化算法
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMin", self.mesh_size / 5.0)

        # 生成 3D 四面体网格
        gmsh.model.mesh.generate(3)
        print("网格生成完毕！")
        
        # ==========================================
        # 提取网格供 fealpy 使用
        # ==========================================
        nodeTags, nodeCoords, _ = gmsh.model.mesh.getNodes()
        nodes = np.array(nodeCoords).reshape(-1, 3)
        tag_to_index = {tag: i for i, tag in enumerate(nodeTags)}
        
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim=3)
        
        tet_cells_tags = None
        for i, eType in enumerate(elemTypes):
            if eType == 4: # 4 代表四面体 (Tetrahedron)
                tet_cells_tags = np.array(elemNodeTags[i]).reshape(-1, 4)
                break
                
        if tet_cells_tags is not None:
            mapper = np.vectorize(lambda tag: tag_to_index[tag])
            cells = mapper(tet_cells_tags)
            try:
                from fealpy.mesh import TetrahedronMesh
                mesh = TetrahedronMesh(nodes, cells)
                print(f"成功为 Fealpy 提取了 {len(cells)} 个加密四面体单元。")
                return mesh
            except ImportError:
                print("未检测到 fealpy 库，跳过 TetrahedronMesh 构建。")
                return None
        else:
            print("警告: 无法提取网格！")
            return None

    def export_mesh(self, filename="bend_pipe_90_native_rotated.msh"):
        """导出网格文件"""
        gmsh.write(filename)
        print(f"网格已成功导出至: {filename}")

    def show_gui(self):
        """打开 Gmsh 图形界面"""
        gmsh.fltk.run()

    def finalize(self):
        """结束 Gmsh 进程"""
        gmsh.finalize()


if __name__ == "__main__":
    geom = PipeGeometry()
    geom.build()

    # 此处接口已修正，采用 bl_size (首层/最小尺寸) 和 bl_thickness (过渡带厚度)
    mesher = PipeMesh(
        geom, 
        mesh_size=0.3,
        bl_enable=True,        
        bl_size=0.02,         # 贴壁处网格细分到 0.02
        bl_thickness=0.15     # 离壁面 0.15 距离后，网格大小恢复到 0.3
    )
    
    mesher.generate_mesh()
    mesher.export_mesh("bend_pipe_benchmark_with_bl.msh")
    mesher.show_gui()
    mesher.finalize()