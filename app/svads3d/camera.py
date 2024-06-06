import numpy as np
from typing import Union


class Camera():
    """
    相机对象，记录相机的位置与朝向，是构造相机系统的基础。
    Attributes:
        picture (Picture): 相机对应的图像。
        location (array): 相机的空间位置（世界坐标）。
        direction (array): 相机的朝向（角度）。
        camear_system (CameraSystem): 相机所处的相机系统。
        mesh: 相机上的网格。
    """
    picture: Picture = None
    location: np.ndarray = None
    direction: np.ndarray = None
    camear_system: CameraSystem = None
    mesh = None

    def __init__(self, picture: Picture, location: np.ndarray, direction: np.ndarray):
        self.picture = picture
        self.picture.camera = self
        self.location = location
        self.direction = direction
        pass

    def to_camera_system(self, *args):
        """
        调和映射，将相机上的点或网格映射到相机系统（视点）。
        @param args: 相机上的点或网格。
        @return:
        """
        assert self.camear_system is not None, "当前相机所属的相机系统未初始化。"
        if type(args[0]) in [list[np.ndarray], np.ndarray]:
            pass
        else:
            pass

    def to_picture(self, *args):
        """
        将相机上的点或网格映射到图像上。
        @param args: 相机上的点或网格
        @return:
        """
        if type(args[0]) in [list[np.ndarray], np.ndarray]:
            pass
        else:
            pass

