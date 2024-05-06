import numpy as np
import os
import cv2
from .ocam_model import OCAMModel
from fealpy.mesh import TriangleMesh

class OCAMSystem:
    def __init__(self, data):
        self.cams = []
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
                radius=data['radius'][i]
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
            'LR', 'LR', 'LR', 'LR', 'LR', 'LR'
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
            [553.866, 992.559],
            [566.843, 986.667],
            [544.920, 978.480],
            [520.528, 985.73],
            [528.981, 961.628],
            [538.6615, 940.2435],
            ], dtype=np.float64)

        radius = np.array([877.5,882.056,886.9275,884.204,883.616,884.5365],dtype=np.float64)
        
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
            'radius' : radius
        }

        return cls(data)


