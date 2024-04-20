import numpy as np

class LevelSet:
    def __init__(self, nelx, nely, domain):
        """
        初始化 LevelSet 类的实例

        Parameters:
        - nelx (int): 结构在 x 方向上的单元数
        - nely (int): 结构在 y 方向上的单元数
        - domain (list): 设计区域的大小
        """
        self._nelx = nelx
        self._nely = nely
        self._domain = domain


    def reinit(self, struc):
        """
        根据给定的结构重置化水平集函数

        该函数通过添加 void 单元的边界来扩展输入结构，计算到最近的 solid 和 void 单元
        的欧几里得距离，并计算水平集函数，该函数在 solid phase 内为正，在 void phase 中为负

        Parameters:
        - struc ( ndarray - (nely, nelx) ): 表示结构的 solid(1) 和 void(0) 单元

        Returns:
        - lsf ( ndarray - (nely+2, nelx+2) ): 表示重置化后的水平集函数
        """
        from scipy import ndimage

        nely, nelx = struc.shape

        # 扩展输入结构，增加边界层，边界上设为 0
        strucFull = np.zeros((nely + 2, nelx + 2))
        strucFull[1:-1, 1:-1] = struc

        # 计算每个网格点到最近的 void (0-值) 单元的距离
        dist_to_0 = ndimage.distance_transform_edt(strucFull)

        # 计算每个网格点到最近的 solid (1-值) 单元的距离
        dist_to_1 = ndimage.distance_transform_edt(strucFull - 1)

        # 调整距离值，每个距离减 0.5，来确保水平集函数在 void phase 和 solid phase 的边界上为零
        # 也就是在 void phase 内为负，solid phase 内为正
        element_length = self._domain[1] / (2*self._nelx)
        temp_0 = dist_to_0 - element_length
        temp_1 = dist_to_1 - element_length

        # 计算水平集函数，void phase 内为负，solid phase 内为正
        lsf = -(1 - strucFull) * temp_1 + strucFull * temp_0

        return lsf
