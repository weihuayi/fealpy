import os
import numpy as np
import cv2
from typing import Union


class Picture():
    """
        图像对象，记录初始图片信息，对特征信息进行处理，是构造相机对象的基础。

        Attributes:
            data (dict): 相关数据，包括图片信息，
            feature_point (list[array]): 特征点。
            feature_line (list[array]): 特征线。
            camera (Camera): 图像对应的相机对象。
            mesh : 图像上生成的网格。
        """

    fname = None
    mark_board = None
    feature_point: list = None
    feature_line: list = []
    camera = None
    mesh = None

    def __init__(self, data_path, fname, feature_point, pic_folder = None):
        """
        初始化图片信息，处理特征点等特征信息。
        1. 读取图片
        2. 处理特征点
        @param data: 相关数据，包括图片信息。
        """
        self.pic_folder = data_path+pic_folder if pic_folder is not None else None 
        self.data_path = data_path
        self.fname = data_path+fname
        # 读取图像
        image = cv2.imread(self.fname, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError
        else:
            self.image = image
        self.height, self.width = self.image.shape

        self.feature_point = feature_point
        self.center, self.radius = self.get_center_and_radius()
        self.rho2theta = None


    def add_feature_point(self, feature_point: Union[np.ndarray, list]):
        """
        手动添加特征点。
        @param feature_point: 特征点。
        @return:
        """
        if isinstance(feature_point, list):
            self.feature_point.extend(feature_point)
        else:
            self.feature_point.append(feature_point)

    def meshing(self, mesh_type):
        """
        根据网格化方案，已有特征点[特征线]生成图像网格。
        @param mesh_type: 图像对应的相机对象。
        @return:
        """
        raise NotImplemented
        self.mesh = None

    def to_camera(self, *args):
        """
        将图像上的点或网格映射到自己对应的相机对象。
        @param args[0]: 需要被映射的点或网格对象。
        @param args[1]: 映射方式。
        @return: 映射后的相机坐标系下的点坐标。
        """
        assert self.camera is not None, "相机未初始化！"
        if type(args[0]) in [np.ndarray, list]:
            point = np.array(args[0])
            NN = len(point)

            fx = self.camera.K[0, 0]
            fy = self.camera.K[1, 1]
            u0 = self.camera.K[0, 2]
            v0 = self.camera.K[1, 2]
            u0, v0 = self.center
            node = np.zeros((NN, 3), dtype=np.float64)

            node[:, 0] = point[:, 0] - u0
            node[:, 1] = point[:, 1] - v0

            phi = np.arctan2(fx * node[:, 1], (fy * node[:, 0]))
            phi[phi < 0] = phi[phi < 0] + np.pi

            idx = np.abs(fx * np.cos(phi)) > 1e-10
            rho = np.zeros_like(phi)
            rho[idx] = node[idx, 0] / (fx * np.cos(phi[idx]))
            rho[~idx] = node[~idx, 1] / (fy * np.sin(phi[~idx]))

            if args[1] == 'L':
                theta = rho
                theta = np.array([self.rho2theta(r) for r in rho])

            node[:, 0] = np.sin(theta) * np.cos(phi)
            node[:, 1] = np.sin(theta) * np.sin(phi)
            node[:, 2] = np.cos(theta)
            return node
        else:
            raise NotImplemented

    def normalizd_coordinate(self, points):
        uv = np.zeros_like(points)
        uv[:, 0] = points[:, 0]/self.width
        uv[:, 1] = points[:, 1]/self.height
        return uv

    def get_center_and_radius(self):
        image = self.image
        # 对图像进行阈值处理，保留非黑色区域
        _, thresholded = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)

        # 找到非黑色区域的轮廓
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 寻找最大轮廓
        max_contour = max(contours, key=cv2.contourArea)

        # 使用最小外接圆找到中心和半径
        center, radius = cv2.minEnclosingCircle(max_contour)

        return center, radius
