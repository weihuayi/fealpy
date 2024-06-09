import numpy as np
import gmsh
from typing import Union
from camera_system import CameraSystem
from .harmonic_map import HarmonicMapData, sphere_harmonic_map

@dataclass
class GroundMarkPoint:
    """
    @brief 地面标记点
    @param points0: 点在相机 cam0 的图像中的坐标
    @param points1: 点在相机 cam1 的图像中的坐标
    @param points2: 点的真实坐标
    """
    cam0:    int
    cam1:    int
    points0: np.ndarray
    points1: np.ndarray
    points2: np.ndarray

class Screen:
    """
    屏幕对象，用于分区，显示，
    Attributes:
        camear_system (CameraSystem): 屏幕对应的相机系统。
        data (dict): 屏幕初始化信息，包括屏幕的长宽高、三个主轴长度、缩放比等数据。
        feature_point (list[array]): 特征点坐标。
        split_line (list[array]): 特征点连成的特征线（分割线）。
        domain (list[array]): 分割线围成分区。
        mesh : 屏幕上生成的网格。
    """
    camear_system: CameraSystem = None
    data: dict = None
    feature_point: Union[np.ndarray, list[np.ndarray]] = None
    split_line: Union[np.ndarray, list[np.ndarray]] = None
    domain: Union[np.ndarray, list[np.ndarray]] = None
    mesh = None

    def __init__(self, camear_system: CameraSystem, data: dict):
        """
        @brief: 屏幕初始化。
                1. 从相机系统获取地面特征点
                2. 优化相机参数
                3. 分区
                4. 网格化 (计算自身的特征点)
                5. 构建相机的框架特征点 (用于调和映射的边界条件)
                6. 计算 uv
        @param feature_point: 屏幕上的特征点，用于网格生成以及计算视点到相机系统的映射。
        """
        self.carsize = None
        self.scale_ratio = None
        self.center_height = None
        self.size = (self.carsize[i]*self.scale_ratio[i] for i in range(3))

        self.camear_system = camear_system
        self.camear_system.screen = self
        self.data = data

        self.gmp = self.ground_mark_board()
        self.meshs, self.didxs, self.dvals = self.meshing()
        self.uvs = self.compute_uv()

    def ground_mark_board(self):
        """
        @brief 获取地面标记点
        """
        gmp = []
        camsys = self.camear_system
        for i in range(6):
            ri = (i+1)%6 # 右边相邻相机
            ps0 = camsys.cams[ i].mark_board[1]
            ps1 = camsys.cams[ri].mark_board[0]
            ps2 = 0.5*camsys.cams[ i].to_screen(ps0, -camsys.center_height)
            ps2+= 0.5*camsys.cams[ri].to_screen(ps1, -camsys.center_height)
            for j in range(len(ps0)):
                gmp.append(GroundMarkPoint(i, ri, ps0[j], ps1[j], ps2[j]))
        return gmp

    def optimize(self):
        """
        相机参数优化方法，根据特征信息优化当前相机系统中的所有相机的位置和角度。
        @param args:
        @return:
        """
        gmp = self.gmp
        align_point = np.array([[g.points0, g.points1] for g in gmp])
        def object_function(self, x):
            """
            @brief The object function to be optimized.
            @param x The parameters to be optimized.
            """
            osysterm = self.ocam_systerm
            models = self.models
            align_point = self.align_point

            osysterm.set_parameters(x.reshape(18,3))

            ## 要对齐的点在屏幕上的坐标
            align_point_screen = np.zeros([6, 2, 12, 3], dtype=np.float_)

            f1, f2 = osysterm.get_implict_surface_function()

            z0 = -osysterm.center_height
            for i, j in itertools.product(range(6), range(2)):
                mod = models[i][j]
                align_point_screen[i, j] = mod.mesh_to_ground(align_point[i, j], z0)

            error = np.sum((align_point_screen[:, 0] - align_point_screen[:, 1])**2)
            return error

    def get_implict_function(self):
        """
        @brief 获取表示自身的隐式函数
        """
        a, b, c = self.size
        z0      = self.center_height
        def f0(p):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            return x**2/a**2 + y**2/b**2 + z**2/c**2 - 1.0
        def f1(p):
            z = p[..., 2]
            return z + z0
        return f0, f1

    def partition(self, partition_type :PartitionType):
        """
        将屏幕区域分区，并通过分区的分割线构造特征点与特征线，可选择 PartitionType 中提供的分区方案。
        @param partition_type: 分区方案。
        @return:
        """
        self.feature_point = None
        self.split_line = None
        self.domain = None
        pass

    def meshing(self, theta = np.pi/6, only_ground=True):#(self, meshing_type:MeshingType):
        """
        在屏幕上生成网格，可选择 MeshingType 中提供的网格化方案。
        @param meshing_type: 网格化方案。
        @return:
        """
        gmsh.initialize()

        gmsh.option.setNumber("Mesh.MeshSizeMax", 1)  # 最大网格尺寸
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)    # 最小网格尺寸

        camsys = self.camear_system

        def add_rectangle(p0, p1, p2, p3):
            """
            @brief 添加矩形
            """
            # 添加线
            l1 = gmsh.model.occ.addLine(p0, p1)
            l2 = gmsh.model.occ.addLine(p1, p2)
            l3 = gmsh.model.occ.addLine(p2, p3)
            l4 = gmsh.model.occ.addLine(p3, p0)
            # 添加 loop
            curve = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
            return gmsh.model.occ.addPlaneSurface([curve])

        # 构造椭球面
        a, b, c = self.size
        z0 = -self.center_height
        phi = np.arcsin(z0 / c)

        # 构造单位球面
        r = 1.0
        ball = gmsh.model.occ.addSphere(0, 0, 0, 1, 1, phi, 0)
        gmsh.model.occ.dilate([(3, ball)],0, 0, 0, a, b, c)
        gmsh.model.occ.remove([(3, ball), (2, 2)])

        # 车辆区域
        vehicle = gmsh.model.occ.addRectangle(-l/2, -w/2, z0, l, w)
        ground = gmsh.model.occ.cut([(2, 1), (2, 3)], [(2, vehicle)])[0]

        v = 30*np.array([[-np.cos(theta), -np.sin(theta)], 
                      [np.cos(theta), -np.sin(theta)], 
                      [np.cos(theta), np.sin(theta)], 
                      [-np.cos(theta), np.sin(theta)]])
        point = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]],
                         dtype=np.float64)
        ps = [3, 4, 5, 6]
        planes  = []
        for i in range(4):
            pp1 = gmsh.model.occ.addPoint(point[i, 0]+v[i, 0], point[i, 1]+v[i, 1], z0)
            pp2 = gmsh.model.occ.addPoint(point[i, 0]+v[i, 0], point[i, 1]+v[i, 1], 0.0)
            pp3 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], 0.0)
            planes.append(add_rectangle(ps[i], pp1, pp2, pp3))

        point = np.array([[0, -w/2], [0, w/2]], dtype=np.float64)
        v = 30*np.array([[0, -1], [0, 1]], dtype=np.float64)
        for i in range(2):
            pp0 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], z0)
            pp1 = gmsh.model.occ.addPoint(point[i, 0]+v[i, 0], point[i, 1]+v[i, 1], z0)
            pp2 = gmsh.model.occ.addPoint(point[i, 0]+v[i, 0], point[i, 1]+v[i, 1], 0.0)
            pp3 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], 0.0)
            planes.append(add_rectangle(pp0, pp1, pp2, pp3))

        # 地面特征点
        gmp = self.gmp
        gmps = [gmsh.model.occ.addPoint(p.points2[0], p.points2[1], p.points2[2]) for p in gmp]
        gmsh.model.occ.fragment(ground, [(0, p) for p in gmps])

        frag = gmsh.model.occ.fragment([(2, 1), (2, 3)], [(2, plane) for plane in planes])
        for i in range(len(frag[1]))[2:]:
            gmsh.model.occ.remove(frag[1][i], recursive=True)

        if only_ground:
            gmsh.model.occ.remove([(2, i+1) for i in range(7)], recursive=True)

        gmsh.model.occ.synchronize()
        #gmsh.fltk.run()

        node = gmsh.model.mesh.getNodes()

        gmsh.model.mesh.generate(2)

        def creat_part_mesh(node, cell):
            idx = np.unique(cell)
            nidxmap = np.zeros(node.shape[0], dtype = np.int_)
            nidxmap[idx] = np.arange(idx.shape[0])
            cell = nidxmap[cell]

            mesh = TriangleMesh(node[idx], cell)
            return mesh, nidxmap

        # 转化为 fealpy 的网格
        node = gmsh.model.mesh.get_nodes()[1].reshape(-1, 3)
        NN = node.shape[0]

        ## 节点编号到标签的映射
        nid2tag = gmsh.model.mesh.get_nodes()[0]
        tag2nid = np.zeros(NN*2, dtype = np.int_)
        tag2nid[nid2tag] = np.arange(NN)

        ## 获取地面特征点在网格中的编号
        gmpidx = np.array([gmsh.model.mesh.get_nodes(0, g)[0] for g in gmps])

        ## 网格分块
        didx = [] # 狄利克雷边界条件的节点编号
        dval = [] # 狄利克雷边界条件的节点值
        partmesh = []
        if only_ground:
            idxs = [[10], [12], [13], [11], [9], [8]]
        else:
            idxs = [[10, 4], [12, 6], [13, 7], [11, 5], [9, 3], [8, 1, 2]]
        for i, idx in enumerate(idxs):
            # 获取第 i 块网格的单元
            cell = []
            for j in idx:
                cell0 = gmsh.model.mesh.get_elements(2, j)[2][0].reshape(-1, 3)
                cell.append(tag2nid[cell0])
            cell = np.concatenate(cell, axis=0)

            f0 = lambda x : camsys.cams[i].project_to_cam_sphere(x)-camsys.cams[i].location
            f1 = lambda x : camsys.cams[i].image_to_camera_sphere(x)-camsys.cams[i].location

            # 获取第 i 块网格的屏幕边界特征点
            geoedge = []
            for j in idx:
                geoedge += gmsh.model.getBoundary([(2, j)])
            geoedge = np.unique([abs(ge[1]) for ge in geoedge])

            lists = [gmsh.model.mesh.get_nodes(1, ge) for ge in geoedge]
            didx_s = [tag2nid[ge[0]] for ge in lists]
            dval_s = [f0(ge[1].reshape(-1, 3)) for ge in lists]

            # 获取第 i 块网格的地面特征点
            ismeshi = np.zeros(NN, dtype = np.bool_)
            ismeshi[cell] = True # 第 i 块网格中的点

            flag = ismeshi[gmpidx]
            didx_g = [gmpidx[j] for j in range(len(flag)) if flag[j]]
            dval_g = [f1(g.points0[None, :]) if g.cam0 == i else
                      f1(g.points1[None, :]) for j, g in enumerate(gmp) if flag[j]]

            pmesh, nidxmap = creat_part_mesh(node, cell)
            partmesh.append(pmesh)
            didx_i = nidxmap[np.concatenate(didx_s+didx_g, dtype=np.int_)]
            dval_i = np.concatenate(dval_s+dval_g, axis=0)
            didx_i = nidxmap[np.concatenate(didx_s, dtype=np.int_)]
            dval_i = np.concatenate(dval_s, axis=0)

            didx_i, uniqueidx = np.unique(didx_i, return_index=True)
            dval_i = dval_i[uniqueidx]
            didx.append(didx_i)
            dval.append(dval_i)

        gmsh.finalize()
        return partmesh, didx, dval 

    def to_view_point(self, points):
        """
        将屏幕上的点或网格映射到视点，同时传递分区、特征信息。
        @param points: 屏幕上的点。
        @return: 映射到相机系统的点。
        """
        return self.camear_system.projecte_to_view_point(points)

    def compute_uv(self):
        """
        计算屏幕上网格点在相机系统中的uv坐标。
        @param args: 屏幕上的点。
        @return:
        """
        uv = []
        for i, cam in enumerate(self.camear_system.cams):
            mesh   = self.meshs[i]
            node_s = mesh.entity('node').copy()
            node   = self.screen_to_viewpoint(node_s)
            mesh.node = node

            data = HarmonicMapData(mesh, self.didx[i], self.dval[i])
            node = sphere_harmonic_map(data).reshape(-1, 3)
            node += cam.location
            uvi = cam.to_picture(node)
            uvi[:, 0] = 1-uvi[:, 0]

            uv.append(uvi)
            mesh.node = node_s
        return uv

    def sphere_to_self(self, points, center, radius):
        """
        将一个球面上的点投影到屏幕上。
        @param points: 球面上的点 (NP, 3)。
        @param center: 球心。
        @param radius: 半径。
        @return: 投影到屏幕上的点。
        """
        f0, f1 = self.get_implict_function()
        ret = np.zeros_like(nodes)

        def sphere_to_implict_function(p, c, r, fun):
            g = lambda t : fun(c + t*(p-c))
            t = fsolve(g, 1000)
            val = c + t*(p-c)
            return val

        g0 = lambda p : sphere_to_implict_function(p, center, radius, f0)
        g1 = lambda p : sphere_to_implict_function(p, center, radius, f1)
        for i, node in enumerate(nodes):
            val = g0(node)
            if val[2] < -self.center_height:
                val = g1(node)
            ret[i] = val
        return ret

    def display(self, plotter):
        """
        显示图像。
        @param plotter: 绘图器。
        """
        uvs = self.uvs
        for uv, mesh, cam in zip(uvs, meshs, cams):
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            no = np.concatenate((node[cell].reshape(-1, 3), uv[cell].reshape(-1, 2)), axis=-1, dtype=np.float32)
            plotter.add_mesh(no, cell=None, texture_path=cam.fname)

