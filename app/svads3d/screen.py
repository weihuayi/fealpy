import numpy as np
from typing import Union


class Screen:
    """
    屏幕对象，用于分区，显示，
    Attributes:
        camear_system (CameraSystem): 屏幕对应的相机系统。
        data (dict): 屏幕初始化信息，包括屏幕的长宽高、三个主轴长度、缩放比等数据。
        feature_point (list[array]): 特征点坐标。
        split_line (list[array]): 特征点连成的特征线（分割线）。
        domain (list[array]): 分割线围成分区。
        mesh : 屏幕上生成的网格。
    """
    camear_system = None
    data: dict = None
    feature_point: Union[np.ndarray, list[np.ndarray]] = None
    split_line: Union[np.ndarray, list[np.ndarray]] = None
    domain: Union[np.ndarray, list[np.ndarray]] = None
    mesh = None

    def __init__(self, camear_system, data: dict):
        """
        @brief: 屏幕初始化。
                1. 从相机系统获取地面特征点
                2. 优化相机参数
                3. 分区
                4. 网格化 (计算自身的特征点)
                5. 构建相机的框架特征点 (用于调和映射的边界条件)
                6. 计算 uv
        @param feature_point: 屏幕上的特征点，用于网格生成以及计算视点到相机系统的映射。
        """
        self.camear_system = camear_system
        self.camear_system.screen = self
        self.data = data
        pass

    def optimize(self, *args):
        """
        相机参数优化方法，根据特征信息优化当前相机系统中的所有相机的位置和角度。
        @param args:
        @return:
        """
        pass


    def partition(self, partition_type):
        """
        将屏幕区域分区，并通过分区的分割线构造特征点与特征线，可选择 PartitionType 中提供的分区方案。
        @param partition_type: 分区方案。
        @return:
        """
        self.feature_point = None
        self.split_line = None
        self.domain = None
        pass

    def meshing(self, meshing_type):
        """
        在屏幕上生成网格，可选择 MeshingType 中提供的网格化方案。
        @param meshing_type: 网格化方案。
        @return:
        """
        pass

    def to_camera_system(self, *args):
        """
        将屏幕上的点或网格映射到相机系统，同时传递分区、特征信息。
        @param args: 屏幕上的点或网格。
        @return:
        """
        if type(args[0]) in [list[np.ndarray], np.ndarray]:
            pass
        else:
            pass

    def compute_uv(self, *args):
        """
        计算屏幕上网格点在相机系统中的uv坐标。
        @param args: 屏幕上的点。
        @return:
        """
        pass

    def display(self, *args):
        """
        显示图像。
        @param args: 相关参数。
        @return:
        """
        pass

    def projecte_to_self(self, *args):
        """
        将相机系统上的点投影到屏幕上。
        @param args: 相机系统上的点。
        @return:
        """
        pass


