import numpy as np
import gmsh
from fealpy.mesh import TriangleMesh
from typing import Union
from camera_system import CameraSystem
from scipy.optimize import fsolve
from harmonic_map import * 
#from COA import COA
#from fealpy.opt import *
from fealpy.opt.optimizer_base import Optimizer, opt_alg_options
from fealpy.opt.crayfish_opt_alg import CrayfishOptAlg
from meshing_type import MeshingType
from partition_type import PartitionType
from dataclasses import dataclass

# 重叠区域网格
@dataclass
class OverlapGroundMesh:
    """
    @brief 重叠地面区域网格
    @param mesh: 网格
    @param cam0: 相机0
    @param cam1: 相机1
    @param uv0: 网格点在相机 0 的图像中的 uv 坐标
    @param uv1: 网格点在相机 1 的图像中的 uv 坐标
    @param w:   网格点上的图像在相机 0 中的权重
    """
    mesh : list = None
    cam0 : int = None
    cam1 : int = None
    uv0  : np.ndarray = None
    uv1  : np.ndarray = None
    w    : np.ndarray = None

@dataclass
class OverlapEllipsoidMesh:
    """
    @brief 重叠椭球区域网格
    @param mesh: 网格
    @param cam0: 相机0
    @param cam1: 相机1
    @param didx0: 相机 0 狄利克雷边界条件的节点编号
    @param didx1: 相机 1 狄利克雷边界条件的节点编号
    @param dval0: 相机 0 狄利克雷边界条件的节点值
    @param dval1: 相机 1 狄利克雷边界条件的节点值
    @param uv0: 网格点在相机 0 的图像中的 uv 坐标
    @param uv1: 网格点在相机 1 的图像中的 uv 坐标
    @param w:   网格点上的图像在相机 0 中的权重
    """
    mesh : list = None
    cam0 : int = None
    cam1 : int = None
    didx0: np.ndarray = None
    didx1: np.ndarray = None
    dval0: np.ndarray = None
    dval1: np.ndarray = None
    uv0  : np.ndarray = None
    uv1  : np.ndarray = None
    w    : np.ndarray = None

@dataclass
class NonOverlapGroundMesh:
    """
    @brief 非重叠地面区域网格
    @param mesh: 网格
    @param cam: 相机
    @param uv: 网格点在相机的图像中的 uv 坐标
    """
    mesh : list = None
    cam  : int = None
    uv   : np.ndarray = None

@dataclass
class NonOverlapEllipsoidMesh:
    """
    @brief 非重叠椭球区域网格
    @param mesh: 网格
    @param cam: 相机
    @param didx: 狄利克雷边界条件的节点编号
    @param dval: 狄利克雷边界条件的节点值
    @param uv: 网格点在相机的图像中的 uv 坐标
    """
    mesh : list = None
    cam  : int = None
    didx : np.ndarray = None
    dval : np.ndarray = None
    uv   : np.ndarray = None

class Screen:
    def __init__(self, camera_system, carsize, scale_ratio, center_height,
                 ptype, mtype=MeshingType.TRIANGLE):
        """
        @brief: 屏幕初始化。
                1. 从相机系统获取地面特征点
                2. 优化相机参数
                3. 分区
                4. 网格化 (计算自身的特征点)
                5. 构建相机的框架特征点 (用于调和映射的边界条件)
                6. 计算 uv
        @param camera_system: 相机系统 
        @param carsize: 车辆尺寸
        @param scale_ratio: 缩放比例
        @param center_height: 车辆中心高度
        @param ptype: 分区类型
        @param mtype: 网格化类型
        """
        self.carsize = carsize
        self.scale_ratio = scale_ratio  
        self.center_height = center_height
        self.axis_length = np.array([self.carsize[i]*self.scale_ratio[i] for i
                                     in range(3)]) # 椭圆轴长

        self.camera_system = camera_system
        self.camera_system.screen = self

        self.partition_type = ptype

        self.ground_overlapmesh = [] 
        self.ground_nonoverlapmesh = [] 
        self.eillposid_overlapmesh = []
        self.eillposid_nonoverlapmesh = []
        
        print("优化相机参数...")
        self.optimize()
        print("优化完成！")
        #self.draw_frature_points()

        print("正在生成网格...")
        self.meshing()
        print("网格生成完成！")

        print("计算网格点参数...")
        self.compute_uv()
        print("计算完成！")


    def get_feature_point(self):
        """
        @brief 获取地面标记点
        """
        gmp = []
        camsys = self.camera_system
        return gmp

    def optimize(self):
        """
        相机参数优化方法，根据特征信息优化当前相机系统中的所有相机的位置和角度。
        """
        camsys = self.camera_system
        self.i=0
        def object_function(x):
            """
            @brief The object function to be optimized.
            @param x The parameters to be optimized.
            """
            N, dim  = x.shape
            error_list = np.zeros((N,))

            for j in range(N):

                x2 = x[j,:].reshape(6, -1)

                camsys.set_parameters(x2)
                error = 0
                for i in range(6):
                    ps0 = camsys.cameras[i].feature_points["ground"]
                    ps1 = camsys.cameras[i].feature_points["camera_sphere"]
                    ps2 = camsys.cameras[i].to_screen(ps1, on_ground=True)
                    ps0 = np.array(ps0)
                    error += np.sum((ps0 - ps2[:, :-1])**2)
                error_list[j] = error 

    

            return error_list.flatten()

        # 6 个相机，每个相机的位置和欧拉角共 6 * 6 = 36 个参数
        init_x = np.zeros((6, 10), dtype=np.float_)
        for i in range(6):
            init_x[i, :3] = camsys.cameras[i].location
            init_x[i, 3:6] = camsys.cameras[i].eular_angle
            init_x[i, 6:] = np.array([1.0, 0.000001, 0.000001, 0.000001])

        #参数设置
        N = 100
        dim = 6*10
        ub = init_x.copy()
        ub[:, 0:3] += 0.1
        ub[:, 3:6] += 0.01
        ub[:, 6:]  += 1
        lb = init_x.copy()
        lb[:, 0:3] -= 0.1
        lb[:, 3:6] -= 0.01
        lb[:, 6]  -= 0.1
        Max_iter = 50

        #opt_alg = COA(N, dim, ub.flatten(), lb.flatten(), Max_iter,
        #              object_function, init_x.flatten())
        
        #best_fitness,best_position,_ = opt_alg.cal()
        
        ub = ub.flatten().reshape((1,dim))
        lb = lb.flatten().reshape((1, dim))
        a = init_x.flatten()[None, :]
        init_x = np.tile(a, (N, 1))
        # init_x = lb + np.random.rand(N, dim) * (ub - lb)
        option = opt_alg_options(init_x, object_function, (lb, ub), N, MaxIters=50)
        optimizer = CrayfishOptAlg(option)

        optimizer.run()
        best_position = optimizer.gbest
        print(best_position)
        camsys.set_parameters(best_position.reshape(6, -1))

    def get_implict_function(self):
        """
        @brief 获取表示自身的隐式函数
        """
        a, b, c = self.axis_length
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

    def partition(self):
        """
        将屏幕区域分区，并通过分区的分割线构造特征点与特征线，可选择 PartitionType 中提供的分区方案。
        """
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
            s = gmsh.model.occ.addPlaneSurface([curve])
            return s

        l, w, h = self.carsize
        a, b, c = self.axis_length
        z0 = -self.center_height

        parameters = self.partition_type.parameters
        # 无重叠分区
        if self.partition_type == "nonoverlap":
            theta = parameters[0]
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

            # 将屏幕分区
            frag = gmsh.model.occ.fragment([(2, 1), (2, 3)], [(2, plane) for plane in planes])
            for i in range(len(frag[1]))[2:]:
                gmsh.model.occ.remove(frag[1][i], recursive=True)

            ground = [[[10], [0]], 
                      [[12], [1]], 
                      [[13], [2]], 
                      [[11], [3]], 
                      [[9], [4]], 
                      [[8], [5]] ]

            eillposid = [[[4], [0]], 
                         [[6], [1]], 
                         [[7], [2]], 
                         [[5], [3]], 
                         [[3], [4]], 
                         [[1, 2], [5]]]

            return ground, eillposid

        # 重叠分区1
        elif self.partition_type == "overlap1":
            theta1 = parameters[0]
            theta2 = parameters[1]
            v1 = 30 * np.array([[-np.cos(theta1), -np.sin(theta1)],
                                [np.cos(theta1), -np.sin(theta1)],
                                [np.cos(theta1), np.sin(theta1)],
                                [-np.cos(theta1), np.sin(theta1)]])
            v2 = 30 * np.array([[-np.cos(theta2), -np.sin(theta2)],
                                [np.cos(theta2), -np.sin(theta2)],
                                [np.cos(theta2), np.sin(theta2)],
                                [-np.cos(theta2), np.sin(theta2)]])
            point = np.array([[-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2]],
                             dtype=np.float64)

            ps = [3, 4, 5, 6]
            planes = []
            for i in range(4):
                pp11 = gmsh.model.occ.addPoint(point[i, 0] + v1[i, 0], point[i, 1] + v1[i, 1], z0)
                pp12 = gmsh.model.occ.addPoint(point[i, 0] + v1[i, 0], point[i, 1] + v1[i, 1], 0.0)
                pp13 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], 0.0)
                planes.append(add_rectangle(ps[i], pp11, pp12, pp13))

                pp21 = gmsh.model.occ.addPoint(point[i, 0] + v2[i, 0], point[i, 1] + v2[i, 1], z0)
                pp22 = gmsh.model.occ.addPoint(point[i, 0] + v2[i, 0], point[i, 1] + v2[i, 1], 0.0)
                planes.append(add_rectangle(ps[i], pp21, pp22, pp13))

            point = np.array([[0, -w / 2], [0, w / 2]], dtype=np.float64)
            v = 30 * np.array([[0, -1], [0, 1]], dtype=np.float64)
            for i in range(len(point)):
                pp0 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], z0)
                pp1 = gmsh.model.occ.addPoint(point[i, 0] + v[i, 0], point[i, 1] + v[i, 1], z0)
                pp2 = gmsh.model.occ.addPoint(point[i, 0] + v[i, 0], point[i, 1] + v[i, 1], 0.0)
                pp3 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], 0.0)
                planes.append(add_rectangle(pp0, pp1, pp2, pp3))

            # 将屏幕分区
            frag = gmsh.model.occ.fragment([(2, 1), (2, 3)], [(2, plane) for plane in planes])
            for i in range(len(frag[1]))[2:]:
                gmsh.model.occ.remove(frag[1][i], recursive=True)

            ground = [[[14], [5, 0, 34, 38]],
                      [[16], [0]],
                      [[18], [1]],
                      [[21], [1, 2, 45, 49]],
                      [[20], [2]],
                      [[19], [2, 3, 47, 44]],
                      [[17], [3]],
                      [[15], [4]],
                      [[13], [4, 5, 37, 36]],
                      [[12], [5]]]

            eillposid = [[[4], [5, 0, 3, 12]],
                         [[6], [0]],
                         [[8], [1]],
                         [[10], [1, 2, 24, 30]],
                         [[11], [2]],
                         [[9], [2, 3, 27, 21]],
                         [[7], [3]],
                         [[5], [4]],
                         [[3], [4, 5, 9, 6]],
                         [[2, 1], [5]]]
            return ground, eillposid
        # 重叠分区2
        elif self.partition_type == "overlap2":
            theta1 = parameters[0]
            theta2 = parameters[1]
            theta3 = parameters[2]
            t = parameters[3]
            v1 = 30 * np.array([[-np.cos(theta1), -np.sin(theta1)],
                                [np.cos(theta1), -np.sin(theta1)],
                                [np.cos(theta1), np.sin(theta1)],
                                [-np.cos(theta1), np.sin(theta1)]])
            v2 = 30 * np.array([[-np.cos(theta2), -np.sin(theta2)],
                                [np.cos(theta2), -np.sin(theta2)],
                                [np.cos(theta2), np.sin(theta2)],
                                [-np.cos(theta2), np.sin(theta2)]])
            point = np.array([[-l / 2, -w / 2], [l / 2, -w / 2], [l / 2, w / 2], [-l / 2, w / 2]],
                             dtype=np.float64)

            ps = [3, 4, 5, 6]
            planes = []
            for i in range(4):
                pp11 = gmsh.model.occ.addPoint(point[i, 0] + v1[i, 0], point[i, 1] + v1[i, 1], z0)
                pp12 = gmsh.model.occ.addPoint(point[i, 0] + v1[i, 0], point[i, 1] + v1[i, 1], 0.0)
                pp13 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], 0.0)
                planes.append(add_rectangle(ps[i], pp11, pp12, pp13))

                pp21 = gmsh.model.occ.addPoint(point[i, 0] + v2[i, 0], point[i, 1] + v2[i, 1], z0)
                pp22 = gmsh.model.occ.addPoint(point[i, 0] + v2[i, 0], point[i, 1] + v2[i, 1], 0.0)
                planes.append(add_rectangle(ps[i], pp21, pp22, pp13))

            point = np.array([[t * l / 2, -w / 2],
                              [-t * l / 2, -w / 2],
                              [-t * l / 2, w / 2],
                              [t * l / 2, w / 2]], dtype=np.float64)
            v = 30 * np.array([[np.cos(theta3), -np.sin(theta3)],
                               [-np.cos(theta3), -np.sin(theta3)],
                               [-np.cos(theta3), np.sin(theta3)],
                               [np.cos(theta3), np.sin(theta3)]], dtype=np.float64)
            for i in range(len(point)):
                pp0 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], z0)
                pp1 = gmsh.model.occ.addPoint(point[i, 0] + v[i, 0], point[i, 1] + v[i, 1], z0)
                pp2 = gmsh.model.occ.addPoint(point[i, 0] + v[i, 0], point[i, 1] + v[i, 1], 0.0)
                pp3 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], 0.0)
                planes.append(add_rectangle(pp0, pp1, pp2, pp3))

            # 将屏幕分区
            frag = gmsh.model.occ.fragment([(2, 1), (2, 3)], [(2, plane) for plane in planes])
            for i in range(len(frag[1]))[2:]:
                gmsh.model.occ.remove(frag[1][i], recursive=True)

            ground = [[[16], [5, 0, 40, 44]],
                      [[18], [0]],
                      [[20], [0, 1, 47, 51]],
                      [[22], [1]],
                      [[25], [1, 2, 55, 59]],
                      [[24], [2]],
                      [[23], [2, 3, 57, 54]],
                      [[21], [3]],
                      [[19], [3, 4, 50, 46]],
                      [[17], [4]],
                      [[15], [4, 5, 43, 42]],
                      [[14], [5]]]

            eillposid = [[[4], [5, 0, 3, 12]],
                         [[6], [0]],
                         [[8], [0, 1, 18, 24]],
                         [[10], [1]],
                         [[12], [1, 2, 30, 36]],
                         [[13], [2]],
                         [[11], [2, 3, 33, 27]],
                         [[9], [3]],
                         [[7], [3, 4, 21, 15]],
                         [[5], [4]],
                         [[3], [4, 5, 9, 6]],
                         [[2, 1], [5]]]
            return ground, eillposid
        else:
            ValueError("Not implemented!")

    def _get_didx_dval(self, mesh, cam, surfaces, tag2nid, nidxmap):
        """
        @brief 获取狄利克雷边界条件的节点编号和值
        """
        cam = self.camera_system.cameras[cam]
        f0 = lambda x : cam.projecte_to_self(x)-cam.location

        # 获取第 i 块网格的屏幕边界特征点
        feature_edge = gmsh.model.getBoundary([(2, j) for j in surfaces])

        # 获取第 i 块网格的屏幕边界特征点的编号和值
        lists = [gmsh.model.mesh.get_nodes(1, abs(ge[1])) for ge in feature_edge]
        didx = [tag2nid[ge[0]] for ge in lists] 
        dval = [f0(ge[1].reshape(-1, 3)) for ge in lists]
        didx = nidxmap[np.concatenate(didx, dtype=np.int_)]
        dval = np.concatenate(dval, axis=0)

        didx, uniqueidx = np.unique(didx, return_index=True)
        dval = dval[uniqueidx]
        return didx, dval

    def _partmeshing(self, surfaces, node, tag2nid, cams, pmesh, overlap = False, is_eillposid = False):
        """
        @brief 生成分区网格
        @param surfaces: 分区面列表
        @param tag2nid: 节点标签到节点编号的映射
        """
        cell = [gmsh.model.mesh.get_elements(2, j)[2][0] for j in surfaces]
        cell = np.concatenate(cell).reshape(-1, 3)
        cell = tag2nid[cell]

        idx = np.unique(cell)
        nidxmap = np.zeros(node.shape[0], dtype = np.int_)
        nidxmap[idx] = np.arange(idx.shape[0])
        cell = nidxmap[cell]

        pmesh.mesh = TriangleMesh(node[idx], cell)

        if is_eillposid&overlap:
            pmesh.didx0, pmesh.dval0 = self._get_didx_dval(pmesh.mesh, cams[0], surfaces, tag2nid, nidxmap)
            pmesh.didx1, pmesh.dval1 = self._get_didx_dval(pmesh.mesh, cams[1], surfaces, tag2nid, nidxmap)
        elif is_eillposid & (not overlap):
            pmesh.didx, pmesh.dval = self._get_didx_dval(pmesh.mesh, cams[0], surfaces, tag2nid, nidxmap)

    def _get_weight(self, node, l0, l1):
        """
        @brief 根据一个点到两个曲线的距离比例计算权重
        """
        p0 = gmsh.model.get_closest_point(1, l0, node)[0]
        p1 = gmsh.model.get_closest_point(1, l1, node)[0]
        l0 = np.linalg.norm(p0-node)
        l1 = np.linalg.norm(p1-node)
        if l0+l1<1e-14:
            return 0.5
        return l0/(l0+l1)

    def meshing(self):
        """
        在屏幕上生成网格，可选择 MeshingType 中提供的网格化方案。
        @param meshing_type: 网格化方案。
        @return:
        """
        gmsh.initialize()


        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.25)  # 最大网格尺寸
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.125)    # 最小网格尺寸

        camerasys = self.camera_system

        # 构造椭球面
        l, w, h = self.carsize
        a, b, c = self.axis_length
        z0 = -self.center_height
        phi = np.arcsin(z0 / c)

        # 构造单位球面
        r = 1.0
        ball = gmsh.model.occ.addSphere(0, 0, 0, 1, 1, phi, 0)
        gmsh.model.occ.dilate([(3, ball)],0, 0, 0, a, b, c)
        gmsh.model.occ.remove([(3, ball), (2, 2)])
        ## 车辆区域
        vehicle = gmsh.model.occ.addRectangle(-l/2, -w/2, z0, l, w)
        ## 生成屏幕
        gmsh.model.occ.cut([(2, 1), (2, 3)], [(2, vehicle)])[0]

        # 分区并生成网格
        ground, eillposid = self.partition()
        gmsh.model.occ.synchronize()
        #gmsh.fltk.run()
        gmsh.model.mesh.generate(2)

        ## 获取节点
        node = gmsh.model.mesh.get_nodes()[1].reshape(-1, 3)
        NN = node.shape[0]

        ## 节点编号到标签的映射
        nid2tag = gmsh.model.mesh.get_nodes()[0]
        tag2nid = np.zeros(NN*2, dtype = np.int_)
        tag2nid[nid2tag] = np.arange(NN)

        # 网格分块
        ## 地面区域网格
        for i, val in enumerate(ground):
            surfaces, cam = val
            if len(cam)==4: # 重叠区域 
                pmesh = OverlapGroundMesh()
                pmesh.cam0, pmesh.cam1 = cam[0], cam[1]
                self._partmeshing(surfaces, node, tag2nid, cam, pmesh, overlap=True)
                self.ground_overlapmesh.append(pmesh)
                pmesh.w = np.array([self._get_weight(node, cam[2], cam[3]) for node in pmesh.mesh.entity('node')], dtype=np.float_)
            else:
                pmesh = NonOverlapGroundMesh()
                pmesh.cam = cam[0]
                self._partmeshing(surfaces, node, tag2nid, cam, pmesh)
                self.ground_nonoverlapmesh.append(pmesh)

        ## 椭球区域网格
        for i, val in enumerate(eillposid):
            surfaces, cam = val
            if len(cam)==4:
                pmesh = OverlapEllipsoidMesh()
                pmesh.cam0, pmesh.cam1 = cam[0], cam[1]
                self._partmeshing(surfaces, node, tag2nid, cam, pmesh, overlap=True, is_eillposid=True)
                self.eillposid_overlapmesh.append(pmesh)
                pmesh.w = np.array([self._get_weight(node, cam[2], cam[3]) for node in pmesh.mesh.entity('node')], dtype=np.float_)
            else:
                pmesh = NonOverlapEllipsoidMesh()
                pmesh.cam = cam[0]
                self._partmeshing(surfaces, node, tag2nid, cam, pmesh, is_eillposid=True)
                self.eillposid_nonoverlapmesh.append(pmesh)
        gmsh.finalize()

    def to_view_point(self, points):
        """
        将屏幕上的点或网格映射到视点，同时传递分区、特征信息。
        @param points: 屏幕上的点。
        @return: 映射到相机系统的点。
        """
        return self.camera_system.projecte_to_view_point(points)

    def _sphere_to_ground(self, points, center, radius):
        """
        将一个球面上的点投影到地面屏幕上。
        @param points: 球面上的点 (NP, 3)。
        @param center: 球心。
        @param radius: 半径。
        @return: 投影到地面屏幕上的点。
        """
        z0 = -self.center_height
        v = points - center[None, :]
        t = (z0-center[2])/v[:, 2]
        val = center + t[:, None]*v
        return val

    def sphere_to_self(self, points, center, radius, on_ground=False):
        """
        将一个球面上的点投影到屏幕上。
        @param points: 球面上的点 (NP, 3)。
        @param center: 球心。
        @param radius: 半径。
        @return: 投影到屏幕上的点。
        """
        if on_ground:
            return self._sphere_to_ground(points, center, radius)

        if len(points.shape)==1:
            points = points[None, :]
        f0, f1 = self.get_implict_function()
        ret = np.zeros_like(points)
        def sphere_to_implict_function(p, c, r, fun):
            g = lambda t : fun(c + t*(p-c))
            t = fsolve(g, 100)
            val = c + t*(p-c)
            return val

        g0 = lambda p : sphere_to_implict_function(p, center, radius, f0)
        g1 = lambda p : sphere_to_implict_function(p, center, radius, f1)
        for i, node in enumerate(points):
            val = g1(node)
            if (f0(val) > 0):
                val = g0(node)
            ret[i] = val
        return ret

    def _compute_uv_of_eillposid(self, mesh, cam, didx, dval):
        """
        @brief 计算屏幕上网格点在相机系统中的uv坐标(调和映射)。
        """
        cam = self.camera_system.cameras[cam]
        node_s = mesh.entity('node').copy()
        node   = self.to_view_point(node_s)
        mesh.node = node

        data = HarmonicMapData(mesh, didx, dval)
        node = sphere_harmonic_map(data).reshape(-1, 3)
        node += cam.location
        uv = cam.to_picture(node, normalizd=True)
        uv[:, 0] = 1-uv[:, 0]
        mesh.node = node_s
        return uv

    def _compute_uv_of_ground(self, mesh, cam):
        """
        @brief 计算屏幕上网格点在相机系统中的uv坐标(无调和映射)。
        """
        cam = self.camera_system.cameras[cam]
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        uv = cam.projecte_to_self(node)
        uv = cam.to_picture(uv, normalizd=True)
        uv[:, 0] = 1-uv[:, 0]
        return uv

    def compute_uv(self):
        """
        @brief 计算网格点在相机系统中的uv坐标。
        """
        # 计算重叠椭球区域网格的uv坐标
        for mesh in self.eillposid_overlapmesh:
            mesh.uv0 = self._compute_uv_of_eillposid(mesh.mesh, mesh.cam0, mesh.didx0, mesh.dval0)
            mesh.uv1 = self._compute_uv_of_eillposid(mesh.mesh, mesh.cam1, mesh.didx1, mesh.dval1)

        # 计算非重叠椭球区域网格的uv坐标
        for mesh in self.eillposid_nonoverlapmesh:
            mesh.uv = self._compute_uv_of_eillposid(mesh.mesh, mesh.cam, mesh.didx, mesh.dval)

        # 计算重叠地面区域网格的uv坐标
        for mesh in self.ground_overlapmesh:
            mesh.uv0 = self._compute_uv_of_ground(mesh.mesh, mesh.cam0)
            mesh.uv1 = self._compute_uv_of_ground(mesh.mesh, mesh.cam1)

        # 计算非重叠地面区域网格的uv坐标
        for mesh in self.ground_nonoverlapmesh:
            mesh.uv = self._compute_uv_of_ground(mesh.mesh, mesh.cam)

    def _display_mesh(self, plotter, pmesh):
        cam = self.camera_system.cameras[pmesh.cam]
        node = pmesh.mesh.entity('node')
        cell = pmesh.mesh.entity('cell')
        no = np.concatenate((node[cell].reshape(-1, 3), pmesh.uv[cell].reshape(-1, 2)), axis=-1, dtype=np.float32)
        plotter.add_mesh(no, cell=None, texture_paths=[cam.picture.fname], 
                         texture_folders=[cam.picture.pic_folder])

    def _display_overlap_mesh(self, plotter, mesh):
        cam0 = self.camera_system.cameras[mesh.cam0]
        cam1 = self.camera_system.cameras[mesh.cam1]
        cell = mesh.mesh.entity('cell')
        node = mesh.mesh.entity('node')[cell].reshape(-1, 3)
        uv0 = mesh.uv0[cell].reshape(-1, 2)
        uv1 = mesh.uv1[cell].reshape(-1, 2)
        w   = mesh.w[cell].reshape(-1, 1)
        has_inf = np.any(np.isinf(w))

        no = np.concatenate((node, uv0, uv1, w), axis=-1, dtype=np.float32)
        plotter.add_mesh(no, cell=None, 
            texture_paths = [cam0.picture.fname, cam1.picture.fname], 
            texture_folders = [cam0.picture.pic_folder, cam1.picture.pic_folder])

    def display(self, plotter):
        """
        显示图像。
        @param plotter: 绘图器。
        """
        cameras = self.camera_system.cameras
        # 非重叠地面区域网格
        for mesh in self.ground_nonoverlapmesh:
            self._display_mesh(plotter, mesh)

        # 非重叠椭球区域网格
        for mesh in self.eillposid_nonoverlapmesh:
            self._display_mesh(plotter, mesh)

        # 重叠地面区域网格
        for mesh in self.ground_overlapmesh:
            self._display_overlap_mesh(plotter, mesh)

        # 重叠椭球区域网格
        for mesh in self.eillposid_overlapmesh:
            self._display_overlap_mesh(plotter, mesh)

    def draw_frature_points(self):
        """
        绘制地面区域。
        @param plotter: 绘图器。
        """
        import matplotlib.pyplot as plt
        cams = self.camera_system.cameras
        for i in range(6):
            ps0 = cams[i].feature_points["ground"]
            ps1 = cams[i].feature_points["camera_sphere"]
            ps2 = cams[i].to_screen(ps1)
            fig = plt.figure()
            ps0 = np.array(ps0)
            # 绘制地面特征点并加上编号
            ax = fig.add_subplot(111)
            ax.scatter(ps0[:, 0], ps0[:, 1], c='r', marker='o')
            for j in range(len(ps0)):
                ax.text(ps0[j, 0], ps0[j, 1], str(j), fontsize=12)

            # 绘制相机球面特征点并加上编号
            ax.scatter(ps2[:, 0], ps2[:, 1], c='b', marker='o')
            for j in range(len(ps2)):
                ax.text(ps2[j, 0], ps2[j, 1], str(j), fontsize=12)
        plt.show()

