import numpy as np
from typing import Union
from harmonic_map import * 


class CameraSystem():
    """
    相机系统对象，用于组装相机，是构建屏幕对象的基础。
    Attributes:
        location (array): 相机系统（视点）的空间位置。
        cameras (list[Camera]): 构成相机系统的相机列表。
        screen (Screen): 相机系统对应的屏幕对象。
        mesh : 相机系统上的网格。
    """
    location: np.ndarray = None
    cameras = None
    screen = None
    mesh = None

    def set_parameters(self, parameters):
        """
        设置相机系统的参数。
        @param parameters: 相机系统的参数。
        """
        for i in range(6):
            loc = parameters[i, :3]
            ang = parameters[i, 3:6]
            coe = parameters[i, 6:]
            self.cameras[i].set_parameters(loc, ang, coe)

    def __init__(self, cameras, viewpoint):
        """
        初始化相机系统对象。
        @param cameras: 构成相机系统的相机列表。
        @param viewpoint: 相机系统的视点。
        """
        self.cameras = cameras
        self.viewpoint = np.array(viewpoint)
        for camera in self.cameras:
            camera.camera_system = self
        self.screen = None

    def to_screen(self, points):
        """
        将相机系统的点映射到屏幕上。
        @param points: 相机系统的点。
        @return: 映射后的屏幕上的点。
        """
        return screen.sphere_to_self(points, self.viewpoint, 1.0)

    def to_camera(self, mesh, didx, dval):
        """
        调和映射，将当前相机系统的网格映射到相机上。
        @param mesh: 网格
        @param didx: 网格狄利克雷点索引
        @param dval: 网格狄利克雷点值
        @return: 映射后的网格点
        """
        data = HarmonicMapData(mesh, didx, dval)
        node = shpere_harmonic_map(data).reshape(-1, 3)
        return node

    def projecte_to_view_point(self, points):
        """
        将点投影到视点上。
        @param points: 要投影的点。
        @return: 投影后的点。
        """
        v = points - self.viewpoint
        v = v / np.linalg.norm(v, axis=-1, keepdims=True)
        return v + self.viewpoint

    def assemble(self):
        """
        将当前相机系统的相机图像组合到相机系统中，并进行相关的剪切、拼接操作。
        @return:
        """
        self.mesh = None
        pass


