import os
import numpy as np
from typing import Union
from .camera import Camera


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

    data: dict = None
    feature_point: Union[list[np.ndarray], np.ndarray] = None
    feature_line: Union[list[np.ndarray], np.ndarray] = None
    camera: Camera = None
    mesh = None

    def __init__(self, data: dict):
        """
        初始化图片信息，处理特征点等特征信息。
        1. 读取图片
        2. 处理特征点
        @param data: 相关数据，包括图片信息。
        """
        self.data = data
        self.feature_point = None
        self.feature_line = None
        pass

    def add_feature_point(self, feature_point: Union[list[np.ndarray], np.ndarray]):
        """
        手动添加特征点。
        @param feature_point: 特征点。
        @return:
        """
        pass

    def meshing(self, mesh_type: MeshingType):
        """
        根据网格化方案，已有特征点[特征线]生成图像网格。
        @param mesh_type: 图像对应的相机对象。
        @return:
        """
        self.mesh = None
        pass

    def to_camera(self, *args):
        """
        将图像上的点或网格映射到自己对应的相机对象。
        @param args: 需要被映射的点或网格。
        @return:
        """
        assert self.camera is not None, "相机未初始化！"
        if type(args[0]) in [list[np.ndarray], np.ndarray]:
            pass
        else:
            pass

