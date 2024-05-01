import numpy as np
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
                flip = data['flip'][i]
            ))

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
            no[:, 1] *= -1.0
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
            no[:, 1] *= -1.0
            print(i, cam.fname)
            uv = cam.world_to_image(no)
            no = np.concatenate((no, uv), axis=-1, dtype=np.float32)
            plotter.add_mesh(no, cell=None, texture_path=cam.fname, flip=cam.flip)
            i0 += 10
            i1 += 10

        # 卡车区域的贴图
        ce = cell[domain == 0]
        no = np.array(node[ce].reshape(-1, node.shape[-1]), dtype=np.float32)
        plotter.add_mesh(no, cell=None, texture_path=None)

    def ellipsoid_mesh_1(self, plotter):
        mesh= TriangleMesh.from_section_ellipsoid(
            size=(17.5, 3.47, 3),
            center_height=3,
            scale_ratio=(1.618, 1.618, 1.618),
            density=0.1,
            top_section=np.pi / 2,
            return_edge=False)

        mesh = TriangleMesh.from_ellipsoid_surface(ntheta=80, nphi=80,
                               radius=(1.618*17.5, 1.618*3.47, 1.618*3.0),
                               theta=(np.pi / 4, 3 * np.pi / 4),
                               )

        node = mesh.entity('node')
        cell = mesh.entity('cell')

    def undistort_cv(self):
        import cv2
        images = []
        for cam in self.cams:
            h = cam.height
            w = cam.width


