import unittest

import numpy as np

from level_set import LevelSet

class TestLevelSet(unittest.TestCase):
    def test_reinit(self):
        # 测试参数
        nelx, nely, domain = 32, 20, [0, 32, 0, 20]

        # 创建 LevelSet 实例
        ls = LevelSet(nelx, nely, domain)

        # 创建一个测试用的结构数组
        struc = np.ones((nely, nelx))
        struc[8:12, 13:19] = 0
        strucFull = np.zeros((nely + 2, nelx + 2))
        strucFull[1:-1, 1:-1] = struc

        # 调用 reinit 函数
        lsf = ls.reinit(struc)

        # 检查输出的尺寸是否正确
        expected_shape = (nely + 2, nelx + 2)
        self.assertEqual(lsf.shape, expected_shape, "Level set function shape is incorrect.")

        # 边界上的值应该接近零（取决于网格尺寸和精度）
        # 检查水平集函数在边界上的值
        self.assertAlmostEqual(lsf[10, 14], 0, delta=0.5, msg="Value on boundary should be near zero")

        # 检查材料区域（solid）的值为正
        self.assertTrue(lsf[10, 7] > 0, "Inside values should be positive")

        # 检查空隙区域（void）的值为负
        self.assertTrue(lsf[10, 17] < 0, "Outside values should be negative")

if __name__ == '__main__':
    unittest.main()
