import numpy as np
import os
import cv2
from .ocam_model import OCAMModel
from fealpy.mesh import TriangleMesh

class OCAMSystem:
    def __init__(self, data):
        self.cams = []
        cps = self.get_split_point()
        for i in range(data['nc']):
            axes = np.zeros((3, 3), dtype=np.float64)
            axes[0, :] = data['axes'][0][i]
            axes[1, :] = data['axes'][1][i]
            axes[2, :] = data['axes'][2][i]
            self.cams.append(OCAMModel(
                location = data['location'][i],
                axes = axes,
                center = data['center'][i],
                height = data['height'],
                width = data['width'],
                ss = np.array(data['ss'][i]),
                pol = np.array(data['pol'][i]),
                affine = data['affine'][i],
                fname = data['fname'][i],
                flip = data['flip'][i],
                chessboardpath=data['chessboardpath'][i],
                icenter=data['icenter'][i],
                radius=data['radius'][i],
                mark_board=data['mark_board'][i],
                camera_points = cps[i]
            ))

    def undistort_cv(self):
        for i, cam in enumerate(self.cams):
            outname = cam.fname[:-4]
            # 进行透视矫正
            result = cam.undistort_chess(cam.fname)
            result = cam.perspective(result)
            cv2.imwrite(outname+'_c.jpg', result)

    def show_parameters(self):
        for i, cam in enumerate(self.cams):
            print(i, "-th camara:")
            print("DIM:\n", cam.DIM)
            print("K:\n", cam.K)
            print("D:\n", cam.D)

    def show_images(self):
        import matplotlib.pyplot as plt
        from PIL import Image
        images = []
        for cam in self.cams:
            images.append(Image.open(cam.fname))

        # 设置显示窗口
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))

        # 使用索引去除空的维度，确保每个 subplot 都用图片填充
        for i, ax in enumerate(axs.flat):
            ax.imshow(np.asarray(images[i]))
            ax.axis('off')  # 不显示坐标轴

        #plt.tight_layout()
        plt.show()

    def sphere_mesh(self, plotter):
        """
        @brief 在世界坐标系的相机位置处生成半球网格
        """
        mesh = TriangleMesh.from_unit_sphere_surface(refine=4)
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        bc = mesh.entity_barycenter('cell')
        cell = cell[bc[:, 2] > 0]
        # 相机坐标系下的点
        vertices = np.array(node[cell].reshape(-1, 3), dtype=np.float64)

        for i, cam in enumerate(self.cams):
            # 相机坐标系下的坐标转换为世界坐标系的坐标
            no = np.einsum('ik, kj->ij', vertices, cam.axes) + cam.location
            uv = cam.cam_to_image(vertices)
            no = np.concatenate((no, uv), axis=-1, dtype=np.float32)
            plotter.add_mesh(no, cell=None, texture_path=cam.fname, flip=cam.flip)

    def ellipsoid_mesh(self, plotter):
        """
        @brief 把椭球面环视网格加入 plotter
        """
        mesh= TriangleMesh.from_section_ellipsoid(
            size=(17.5, 3.47, 3),
            center_height=3,
            scale_ratio=(1.618, 1.618, 1.618),
            density=0.1,
            top_section=np.pi / 2,
            return_edge=False)

        node = mesh.entity('node')
        cell = mesh.entity('cell')
        domain = mesh.celldata['domain']

        """
        cd = domain.copy()
        cd[(domain == 11) | (domain == 12)] = domain[(domain == 51) | (domain == 52)]
        cd[(domain == 21) | (domain == 22)] = domain[(domain == 41) | (domain == 42)]
        cd[(domain == 41) | (domain == 42)] = domain[(domain == 21) | (domain == 22)]
        cd[(domain == 51) | (domain == 52)] = domain[(domain == 11) | (domain == 12)]
        domain = cd
        """

        i0, i1 = 11, 12
        for i, cam in enumerate(self.cams):
            ce = cell[(domain == i0) | (domain == i1)]
            no = node[ce].reshape(-1, node.shape[-1])
            uv = cam.world_to_image(no)
            no = np.concatenate((no, uv), axis=-1, dtype=np.float32)
            plotter.add_mesh(no, cell=None, texture_path=cam.fname, flip=cam.flip)
            i0 += 10
            i1 += 10

        # 卡车区域的贴图
        ce = cell[domain == 0]
        no = np.array(node[ce].reshape(-1, node.shape[-1]), dtype=np.float32)
        plotter.add_mesh(no, cell=None, texture_path=None)

    def test_plain_domain(self, plotter, z=10, icam=-1):
        """
        @brief 测试半球外一平面网格
        """
        mesh = TriangleMesh.from_box([-10, 10, -10, 10], nx=100, ny=100)

        NN = mesh.number_of_nodes()
        node = np.zeros((NN, 3), dtype=np.float64)
        node[:, 0:2] = mesh.entity('node')
        node[:, -1] = z
        cell = mesh.entity('cell')

        #投影到球面上
        snode = node / np.sqrt(np.sum(node**2, axis=-1, keepdims=True))
        snode = snode[cell].reshape(-1, node.shape[-1])

        uv = self.cams[icam].cam_to_image(snode)
        pnode = node[cell].reshape(-1, node.shape[-1])
        pnode = np.concatenate((pnode, uv), axis=-1, dtype=np.float32)
        snode = np.concatenate((snode, uv), axis=-1, dtype=np.float32)

        #添加平面网格
        plotter.add_mesh(pnode, cell=None, texture_path=self.cams[icam].fname)

        #添加球面网格
        plotter.add_mesh(snode, cell=None, texture_path=self.cams[icam].fname)

    def test_half_sphere_surface(self, plotter, r=1.0, icam=-1):
        """
        @brief 在单位半球(z > 0)上的三角形网格上进行帖图
        """
        mesh = TriangleMesh.from_unit_sphere_surface(refine=4)
        node = r*mesh.entity('node')
        cell = mesh.entity('cell')
        bc = mesh.entity_barycenter('cell')
        cell = cell[bc[:, 2] > 0]
        # 相机坐标系下的点
        vertices = np.array(node[cell].reshape(-1, 3), dtype=np.float64)

        uv = self.cams[icam].cam_to_image(vertices)
        no = np.concatenate((vertices, uv), axis=-1, dtype=np.float32)
        plotter.add_mesh(no, cell=None, texture_path=self.cams[icam].fname)
        return mesh, uv
    
    def test_half_sphere_surface_with_cutting(self, plotter, theta=np.pi*35/180, h=0.1, icam=-1, ptype='O'):
        """
        """
        mesh = TriangleMesh.from_half_sphere_surface_with_cutting(theta=theta, h=h)
        node = mesh.entity('node')
        cell = mesh.entity('cell')


        # 相机坐标系下的点
        vertices = np.array(node[cell].reshape(-1, 3), dtype=np.float64)
        uv = self.cams[icam].cam_to_image(vertices, ptype=ptype)


        """
        r = np.sin(theta) # 圆柱面半径
        phi = np.arctan2(node[:, 0], node[:, 2])
        phi = phi % (2 * np.pi)
        node[:, 2] = r * np.cos(phi)
        node[:, 0] = r * np.sin(phi)
        vertices = np.array(node[cell].reshape(-1, 3), dtype=np.float64)
        """

        no = np.concatenate((vertices, uv), axis=-1, dtype=np.float32)
        plotter.add_mesh(no, cell=None, texture_path=self.cams[icam].fname)
        return mesh, uv

    def get_split_point(self,
                        size=(17.5, 3.47, 3),
                        scale_ratio=(1.618, 1.618, 1.618),
                        densty=0.05,
                        center_height=3,
                        v=0.5,
                        theta1= 0,
                        theta2= 0):
        '''
        获取分割线
        @param size: 小车长宽高
        @param scale_ratio: 三个主轴的伸缩比例
        @param densty: 节点密度
        @param center_height: 椭球面球心的高度
        @param v: 两侧摄像头分割线相对位置
        @param theta1: 主分割线偏转角
        @param theta2: 两侧分割线
        @return: 分割线上点的笛卡尔坐标
        '''
        import gmsh
        pi = np.pi
        gmsh.initialize()
        l, w, h = size

        # 构造椭球面
        a = l * scale_ratio[0]
        b = w * scale_ratio[1]
        c = h * scale_ratio[2]

        bottom = center_height / c

        # 构造单位球面
        r = 1.0
        ball = gmsh.model.occ.addSphere(0, 0, 0, r, 1, -pi / 2, 0)

        # 底面截取
        box = gmsh.model.occ.addBox(-1, -1, -1, 2, 2, 1 - bottom)
        half_ball = gmsh.model.occ.cut([(3, ball)], [(3, box)])[0]

        # 底部矩形附着
        rec = gmsh.model.occ.addRectangle(-0.5 / scale_ratio[0], -0.5 / scale_ratio[1], -bottom, 1 / scale_ratio[0],
                                          1 / scale_ratio[1])
        gmsh.model.occ.synchronize()
        gmsh.model.mesh.generate(2)
        gmsh.fltk.run()

        # 分割线对应的固定点
        fixed_point1 = [0.5 * r / scale_ratio[0], -0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point2 = [0.5 * r * v / scale_ratio[0], -0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point3 = [-0.5 * r * v / scale_ratio[0], -0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point4 = [-0.5 * r / scale_ratio[0], -0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point5 = [-0.5 * r / scale_ratio[0], 0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point6 = [-0.5 * r * v / scale_ratio[0], 0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point7 = [0.5 * r * v / scale_ratio[0], 0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point8 = [0.5 * r / scale_ratio[0], 0.5 * r / scale_ratio[1], -bottom - 0.5 * r]

        def get_plane(fixed_point, phi):
            p1 = gmsh.model.occ.addPoint(fixed_point[0], fixed_point[1], fixed_point[2])
            p2 = gmsh.model.occ.addPoint(fixed_point[0] + r * np.cos(phi), fixed_point[1] + r * np.sin(phi),
                                         fixed_point[2])
            p3 = gmsh.model.occ.addPoint(fixed_point[0] + r * np.cos(phi), fixed_point[1] + r * np.sin(phi),
                                         fixed_point[2] + 1.5 * r)
            p4 = gmsh.model.occ.addPoint(fixed_point[0], fixed_point[1], fixed_point[2] + 1.5 * r)
            line1 = gmsh.model.occ.addLine(p1, p2)
            line2 = gmsh.model.occ.addLine(p2, p3)
            line3 = gmsh.model.occ.addLine(p3, p4)
            line4 = gmsh.model.occ.addLine(p4, p1)
            curve = gmsh.model.occ.addCurveLoop([line1, line2, line3, line4])
            plane = gmsh.model.occ.addPlaneSurface([curve])

            return plane

        # 分割平面
        plane1 = get_plane(fixed_point1, -theta1)
        plane2 = get_plane(fixed_point1, -pi / 2 + theta1)
        plane3 = get_plane(fixed_point2, -pi / 2 - theta2)
        plane4 = get_plane(fixed_point3, -pi / 2 + theta2)
        plane5 = get_plane(fixed_point4, -pi / 2 - theta1)
        plane6 = get_plane(fixed_point4, -pi + theta1)
        plane7 = get_plane(fixed_point5, pi - theta1)
        plane8 = get_plane(fixed_point5, pi / 2 + theta1)
        plane9 = get_plane(fixed_point6, pi / 2 - theta2)
        plane10 = get_plane(fixed_point7, pi / 2 + theta2)
        plane11 = get_plane(fixed_point8, pi / 2 - theta1)
        plane12 = get_plane(fixed_point8, theta1)

        ov = gmsh.model.occ.fragment(half_ball, [(2, rec)])[0]

        # 分割面与主体求交，得到分割线
        intersection = gmsh.model.occ.intersect(ov,
                                                [(2, plane1), (2, plane2), (2, plane3), (2, plane4), (2, plane5),
                                                 (2, plane6),
                                                 (2, plane7), (2, plane8), (2, plane9), (2, plane10), (2, plane11),
                                                 (2, plane12)])

        gmsh.model.occ.synchronize()
        # 调整网格密度
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), densty)
        gmsh.model.mesh.generate(2)

        # 获取分割线节点
        total_vects = []
        start_surface_tag = 5
        for i in range(12):
            surface_tag = start_surface_tag + i
            curves = gmsh.model.getBoundary([(2, surface_tag)], combined=False, oriented=True)[1:3]
            curve_vects = []
            for curve in curves:
                tag = abs(curve[1])
                node_pre = gmsh.model.mesh.getNodes(1, tag, includeBoundary=True)[1].reshape(-1, 3)
                nodes = np.zeros_like(node_pre)
                nodes[0, :] = node_pre[-2, :]
                nodes[-1, :] = node_pre[-1, :]
                nodes[1:-1, :] = node_pre[0:-2, :]
                # 由球面映射到椭球面
                nodes[:, 0] *= a
                nodes[:, 1] *= b
                nodes[:, 2] *= c
                curve_vects.append(nodes)
            total_vects.append(curve_vects)

        # 显示
        #gmsh.fltk.run()
        camera_points = []
        for i in range(6):
            camera_points.append([])
            camera_points[i].append(total_vects[(2 * i) % 12])
            camera_points[i].append(total_vects[(2 * i + 1) % 12])
            camera_points[i].append(total_vects[(2 * i + 2) % 12])
            camera_points[i].append(total_vects[(2 * i + 3) % 12])
        gmsh.finalize()
        return camera_points

    def undistort_cv(self):
        import cv2
        images = []
        for cam in self.cams:
            h = cam.height
            w = cam.width


    @classmethod
    def from_data(cls, h=3.0):
        """
        @brief 测试数据

        @param[in] h : 世界坐标原点到地面的高度
        """
        location = np.array([ # 相机在世界坐标系中的位置
            [ 8.35/2.0, -3.47/2.0, 1.515], # 右前
            [-8.35/2.0, -3.47/2.0, 1.505], # 右后
            [-17.5/2.0,       0.0, 1.295], # 后
            [-8.35/2.0,  3.47/2.0, 1.495], # 左后
            [ 8.35/2.0,  3.47/2.0, 1.495], # 左前
            [ 17.5/2.0,       0.0, 1.345]  # 前
            ], dtype=np.float64)
        location[:, 2] -= h

        t = np.sqrt(2.0)/2.0
        cz = np.array([ # 相机 z 轴在世界坐标系中的指向
            [0.0,  -t, -t],  # 右前
            [0.0,  -t, -t],  # 右后
            [ -t, 0.0, -t],  # 后 
            [0.0,   t, -t],  # 左后
            [0.0,   t, -t],  # 左前
            [  t, 0.0, -t]   # 前
            ], dtype=np.float64)
        cx = np.array([ # 相机 x 轴在世界坐标系中的指向
            [-1.0,  0.0, 0.0],  # 右前
            [-1.0,  0.0, 0.0],  # 右后
            [ 0.0,  1.0, 0.0],  # 后 
            [ 1.0,  0.0, 0.0],  # 左后
            [ 1.0,  0.0, 0.0],  # 左前
            [ 0.0, -1.0, 0.0]   # 前
            ], dtype=np.float64)
        cy = np.cross(cz, cx) # 相机 y 轴在世界坐标系中的指向

        #polynomial coefficients for the DIRECT mapping function (ocam_model.ss in MATLAB). These are used by cam2world
        ss = [
        [-5.763797e+02, 0.000000e+00, 7.185556e-04, -3.399070e-07, 5.242219e-10],
        [-5.757232e+02, 0.000000e+00, 7.587041e-04, -3.740247e-07, 5.173472e-10],
        [-5.769944e+02, 0.000000e+00, 6.960907e-04, -2.129561e-07, 3.806627e-10],
        [-5.757232e+02, 0.000000e+00, 7.587041e-04, -3.740247e-07, 5.173472e-10],
        [-5.763797e+02, 0.000000e+00, 7.185556e-04, -3.399070e-07, 5.242219e-10],
        [-5.751298e+02, 0.000000e+00, 7.332358e-04, -3.633660e-07, 5.286731e-10]
        ]

        #polynomial coefficients for the inverse mapping function (ocam_model.invpol in MATLAB). These are used by world2cam
        pol = [
            [845.644875, 482.093504, -4.074978, 71.443521, 34.750033, 3.348958, 19.469493, 10.236789, -11.771018, -10.331102, -2.154892],
            [842.618702, 489.883562, 3.551579, 68.390516, 35.533898, -0.486649, 12.653096, 21.865068, 9.894399, 1.351086],
            [853.690706, 511.122043, 22.215504, 72.273914, 36.289875, 7.590651, 14.715128, 16.256317, 6.272230, 0.752288],
            [842.618702, 489.883562, 3.551579, 68.390516, 35.533898, -0.486649, 12.653096, 21.865068, 9.894399, 1.351086],
            [845.644875, 482.093504, -4.074978, 71.443521, 34.750033, 3.348958, 19.469493, 10.236789, -11.771018, -10.331102, -2.154892],
            [845.738193, 486.117526, -3.075807, 69.772397, 36.084962, 2.499655, 17.305930, 12.154529, -8.322921, -8.780900, -1.922651],
            ]

        #center: "row" and "column", starting from 0 (C convention)
        center = np.array([
            [559.875074, 992.836922],
            [575.297515, 987.142409],
            [533.159817, 992.262661],
            [595.297515, 987.142409],
            [559.875074, 982.836922],
            [539.106804, 939.819626],
            ], dtype=np.float64)

        # 仿射系数
        affine = np.array([
            [1.000938,  0.000132, -0.000096],
            [1.000004, -0.000176, -0.000151],
            [1.000921,  0.000077,  0.000329],
            [1.000004, -0.000176, -0.000151],
            [1.000938,  0.000132, -0.000096],
            [1.000375,  0.000070,  0.000432],
            ], dtype=np.float64)

        # 默认文件目录位置
        fname = [
            os.path.expanduser('~/data/src_1.jpg'),
            os.path.expanduser('~/data/src_2.jpg'),
            os.path.expanduser('~/data/src_3.jpg'),
            os.path.expanduser('~/data/src_4.jpg'),
            os.path.expanduser('~/data/src_5.jpg'),
            os.path.expanduser('~/data/src_6.jpg'),
            ]

        flip = [
            None, None, None, None, None, None 
        ]

        chessboardpath = [
            os.path.expanduser('~/data/camera_models/chessboard_2'),
            os.path.expanduser('~/data/camera_models/chessboard_1'),
            os.path.expanduser('~/data/camera_models/chessboard_3'),
            os.path.expanduser('~/data/camera_models/chessboard_4'),
            os.path.expanduser('~/data/camera_models/chessboard_5'),
            os.path.expanduser('~/data/camera_models/chessboard_6'),
            ]

        icenter = np.array([
            [992.559,526.134],
            [986.667,513.157],
            [978.480,535.08],
            [985.73,559.472],
            [961.628,551.019],
            [940.2435,541.3385],
            ], dtype=np.float64)

        radius = np.array([877.5,882.056,886.9275,884.204,883.616,884.5365],dtype=np.float64)
        mark_board = np.array(
        [[(240.946,702.130),(265.611,726.620),(326.989,605.794),(291.346,599.545),
         (248.493,694.023),(268.282,711.997),(314.241,614.862),(288.833,609.140),
         (265.521,668.427),(275.295,673.864),(291.393,641.555),(280.467,638.472),
         (1673.001,772.285),(1711.515,733.990),(1654.236,612.171),(1600.493,622.823),
         (1671.702,754.323),(1702.732,725.841),(1654.219,624.930),(1616.665,634.889),
         (1666.478,705.236),(1680.064,697.104),(1661.974,660.024),(1647.485,665.864)],
         [(268.593,717.706),(308.979,755.603),(384.439,608.193),(329.074,598.007),
         (279.318,709.072),(309.456,737.338),(369.147,619.301),(329.805,611.696),
         (301.373,681.322),(315.767,690.052,),(339.370,650.618),(320.949,644.135),
         (1704.463,768.055),(1732.520,732.221),(1681.072,626.701),(1646.601,638.913),
         (1703.475,749.778),(1722.529,726.180),(1682.977,638.310),(1657.391,648.564),
         (1696.748,710.659),(1706.177,701.772),(1690.590,669.869),(1680.413,675.212)],
         [(336.722,761.334),(459.065,843.159),(561.643,572.304),(421.546,567.267),
         (356.073,747.818),(451.158,806.274),(534.843,593.878),(428.283,584.402),
         (400.842,700.596),(436.129,717.193),(467.284,647.240),(427.526,638.794),
         (1532.010,805.481),(1654.638,732.667),(1560.331,543.526),(1423.584,549.188),
         (1540.329,771.869),(1635.315,721.673),(1557.946,560.666),(1450.886,568.042),
         (1552.074,688.311),(1588.403,675.260),(1557.017,613.734),(1522.558,621.695)],
         [(233.503,674.422),(259.214,702.625),(316.859,578.079),(279.466,569.207),
         (243.090,665.626),(260.053,685.466),(305.689,587.776),(281.988,581.209),
         (258.467,642.556),(268.377,648.475),(284.372,614.911),(272.615,611.244),
         (1666.714,711.499),(1703.773,672.993),(1643.246,554.714),(1594.974,564.733),
         (1665.166,695.445),(1695.455,665.853),(1643.087,566.604),(1606.476,577.471),
         (1658.640,646.255),(1670.754,637.334),(1652.185,600.063),(1637.795,607.301)],
         [(260.645,674.616),(302.728,708.959),(378.877,560.912),(320.521,553.519),
         (270.127,666.156),(303.906,692.382),(364.535,571.969),(321.852,565.919),
         (295.149,637.813),(309.516,644.059),(331.469,605.169),(312.906,599.970),
         (1690.343,688.457),(1716.687,664.436),(1665.597,557.933),(1629.563,564.493),
         (1688.991,673.511),(1708.099,655.327),(1667.512,571.222),(1642.266,576.273),
         (1681.099,637.213),(1691.977,630.243),(1675.108,598.143),(1663.869,602.714)],
         [(266.546,761.228),(368.820,838.300),(471.901,599.210),(350.959,587.009),
         (282.813,752.674),(364.447,809.124),(447.787,614.515),(352.791,602.933),
         (323.830,712.295),(356.378,726.475),(383.816,661.568),(349.424,653.671),
         (1460.643,835.675),(1581.636,748.868),(1483.032,566.232),(1344.212,587.207),
         (1468.790,802.020),(1562.534,736.320),(1480.875,583.683),(1372.462,602.471),
         (1480.763,713.575),(1514.355,696.954),(1483.021,638.530),(1446.418,650.523)]
         ],dtype=np.float64)

        mark_board[...,1] = 1080-mark_board[...,1] 
        data = {
            "nc" : 6,
            "location" : location,
            "axes" : (cx, cy, cz),
            "center" : center,
            "ss" : ss,
            "pol" : pol,
            "affine" : affine,
            "fname" : fname,
            "chessboardpath": chessboardpath,
            "width" : 1920,
            "height" : 1080,
            "vfield" : (110, 180),
            'flip' : flip,
            'icenter': icenter,
            'radius' : radius,
            'mark_board': mark_board
        }

        return cls(data)


