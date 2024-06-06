import numpy as np
from typing import Union


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
    cameras: list[Camera] = None
    screen: Screen = None
    mesh = None

    def __init__(self, location: np.ndarray, cameras: list[Camera]):
        self.location = location
        self.cameras = cameras
        for camera in self.cameras:
            camera.camear_system = self
        pass

    def optimize(self, *args):
        """
        相机参数优化方法，根据特征信息优化当前相机系统中的所有相机的位置和角度。
        @param args:
        @return:
        """
        pass

    def to_screen(self, *args):
        """

        @param args:
        @return:
        """
        assert self.screen is not None, "相机系统所属的屏幕未初始化。"
        if type(args[0]) in [list[np.ndarray], np.ndarray]:
            pass
        else:
            pass

    def to_camera(self, *args):
        """
        调和映射，将当前相机系统的点或网格映射到相机上。
        @param args: 相机系统的点或网格。
        @return:
        """
        if type(args[0]) in [list[np.ndarray], np.ndarray]:
            pass
        else:
            pass

    def assemble(self):
        """
        将当前相机系统的相机图像组合到相机系统中，并进行相关的剪切、拼接操作。
        @return:
        """
        self.mesh = None
        pass


