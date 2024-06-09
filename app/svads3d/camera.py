import numpy as np
from typing import Union
from scipy.optimize import fsolve


class Camera():
    """
    相机对象，记录相机的位置与朝向，是构造相机系统的基础。
    Attributes:
        picture (Picture): 相机对应的图像。
        location (array): 相机的空间位置（世界坐标）。
        eular_angle (array): 相机的欧拉角
        camear_system (CameraSystem): 相机所处的相机系统。
        mesh: 相机上的网格。
    """
    picture: Picture = None
    location: np.ndarray = None
    eular_angle: np.ndarray = None
    camear_system: CameraSystem = None
    mesh = None

    def __init__(self, picture: Picture, location: np.ndarray, eular_angle: np.ndarray):
        """
        @brief 构造函数。
            1. 获取图片到自身的特征点（地面特征点）

        @param picture: 相机对应的图像。
        @param location: 相机的空间位置（世界坐标）。
        @param eular_angle: 相机的朝向（角度）。
        """
        self.picture = picture
        self.picture.camera = self

        self.location = location
        self.eular_angle = eular_angle
        self.axes = self.get_rot_matrix(eular_angle[0], eular_angle[1], eular_angle[2])

        self.system = None
        self.ground_feature_points = None
        self.screen_feature_points = None

    def set_screen_frature_points(self, *args):
        """
        @brief 设置相机的屏幕特征点。
        @param args: 屏幕特征点。
        @return:
        """
        pass

    def get_rot_matrix(self, theta, gamma, beta) -> np.ndarray:
        """
        @brief 从欧拉角计算旋转矩阵。
        @param theta: 绕x轴的旋转角度。
        @param gamma: 绕y轴的旋转角度。
        @param beta: 绕z轴的旋转角度。
        @return: 旋转矩阵。
        """
        # 绕 x 轴的旋转矩阵
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])

        # 绕 y 轴的旋转矩阵
        R_y = np.array([[np.cos(gamma), 0, np.sin(gamma)],
                        [0, 1, 0],
                        [-np.sin(gamma), 0, np.cos(gamma)]])

        # 绕 z 轴的旋转矩阵
        R_z = np.array([[np.cos(beta), -np.sin(beta), 0],
                        [np.sin(beta), np.cos(beta), 0],
                        [0, 0, 1]])
        R = R_z @ R_y @ R_x
        return R

    def to_camera_system(self, *args):
        """
        @brief 调和映射，将相机上的点或网格映射到相机系统（视点）。
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
        @brief 将相机上的点或网格映射到图像上。
        @param args: 相机上的点或网格
        @return:
        """
        if type(args[0]) in [list[np.ndarray], np.ndarray]:
            pass
        else:
            pass

    def projecte_to_self(self, point):
        """
        将点投影到相机球面上。
        @param points: 要投影的点。
        @return: 投影后的点。
        """
        v = points - self.location
        v = v/np.linalg.norm(v, axis=-1, keepdims=True)
        return v + self.location


    def to_screen(self, points):
        """
        将相机球面上的点投影到屏幕上。
        @param args: 相机球面上的点。
        @return:
        """
        screen = self.system.screen
        ret = screen.projecte_to_self(points, self.location, 1.0)
        return ret


