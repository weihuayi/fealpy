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
    camear_system: CameraSystem = None
    data: dict = None
    feature_point: Union[np.ndarray, list[np.ndarray]] = None
    split_line: Union[np.ndarray, list[np.ndarray]] = None
    domain: Union[np.ndarray, list[np.ndarray]] = None
    mesh = None

    def __init__(self, camear_system: CameraSystem, data: dict):
        self.camear_system = camear_system
        self.camear_system.screen = self
        self.data = data
        pass

    def partition(self, partition_type :PartitionType):
        """
        将屏幕区域分区，并通过分区的分割线构造特征点与特征线，可选择 PartitionType 中提供的分区方案。
        @param partition_type: 分区方案。
        @return:
        """
        self.feature_point = None
        self.split_line = None
        self.domain = None
        pass

    def meshing(self, meshing_type:MeshingType):
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

    def display(self, *args):
        """
        显示图像。
        @param args: 相关参数。
        @return:
        """
        pass