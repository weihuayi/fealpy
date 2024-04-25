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

    def sphere_mesh(self):
        mesh = TriangleMesh.from_unit_sphere_surface(refine=3)

        node = mesh.entity('node')
        cell = mesh.entity('cell')

        bc = mesh.entity_barycenter('cell')
        cell = cell[bc[:, 2] > 0]

        vertices = np.array(node[cell].reshape(-1, 3), dtype=np.float32)
        for cam in self.cams:
            uv = cam.cam_to_image(vertices)


    def undistort_cv(self):
        import cv2
        images = []
        for cam in self.cams:
            h = cam.height
            w = cam.width


