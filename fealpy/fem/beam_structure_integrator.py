import numpy as np


class EulerBernoulliBeamStructureIntegrator:
    def __init__(self, E, I, A):
        """
        EulerBermoulliBeamStructureIntegrator 类的初始化

        参数:
        E -- 杨氏模量
        I -- 惯性矩
        A -- 截面积
        """
        self.E = E
        self.I = I
        self.A = A

    def assembly_cell_matrix(self, space, index = np.s_[:], 
                            cellmeasure = None, out = None):
        """
        组装单元刚度矩阵

        参数:
        space -- 空间维度的元组
        index -- 单元网格的索引，默认为全部单元
        cellmeasure -- 单元的度量，默认为 None，表示使用默认度量
        out -- 输出的数组，默认为 None，表示创建新数组

        返回值:
        组装好的单元网格刚度矩阵
        """
        assert isinstance(space, tuple) 
        space0 = space[0]
        mesh = space0.mesh
        GD = mesh.geo_dimension()

        assert  len(space) == 3

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        l = cellmeasure
        c0 = self.E*self.A
        c1 = self.E*self.I
        NC = len(cellmeasure)
        ldof = 3 # 1 element has 3 dof

        if out is None:
            K = np.zeros((NC, GD*ldof, GD*ldof), dtype=np.float64)
        else:
            assert out.shape == (NC, GD*ldof, GD*ldof)
            K = out

        K0 = c0/(2*l)[:, np.newaxis, np.newaxis] * np.array([[1, -1], [-1, 1]])
        print("轴向刚度矩阵 K0:", K0.shape)
        # print("K0:\n", K0)

        K1 = np.zeros((NC, GD*2, GD*2), dtype=np.float64)

        K1[:, 0, 1] = 6 * l
        K1[:, 1, 0] = 6 * l
        K1[:, 1, 1] = 4 * l**2
        K1[:, 1, 2] = -6 * l
        K1[:, 1, 3] = 2 * l**2
        K1[:, 2, 1] = -6 * l
        K1[:, 3, 0] = 6 * l
        K1[:, 3, 1] = 2 * l**2
        K1[:, 3, 2] = -6 * l
        K1[:, 3, 3] = 4 * l**2

        K1[:, 0, 0] = 12
        K1[:, 0, 2] = -12
        K1[:, 0, 3] = 6 * l
        K1[:, 2, 0] = -12
        K1[:, 2, 2] = 12
        K1[:, 2, 3] = -6 * l

        K1 *= (c1/l**3)[:, np.newaxis, np.newaxis]

        print("弯曲刚度矩阵 K1:", K1.shape)
        # print("K1:\n", K1)

        # 使用 K0 填充对应的位置
        K[:, 0, 0] = K0[:, 0, 0]
        K[:, 0, 3] = K0[:, 0, 1]
        K[:, 3, 0] = K0[:, 1, 0]
        K[:, 3, 3] = K0[:, 1, 1]

        # 使用 K1 填充对应的位置
        K[:, 1:3, 1:3] = K1[:, 0:2, 0:2]
        K[:, 1:3, 4:6] = K1[:, 0:2, 2:4]
        K[:, 4:6, 1:3] = K1[:, 2:4, 0:2]
        K[:, 4:6, 4:6] = K1[:, 2:4, 2:4]

        print("局部坐标系下的单元刚度矩阵 K_shape:", K.shape)
        # print("K:\n", K)

        tan = mesh.cell_unit_tangent(index=index) # 计算单元的单位切向矢量（即轴线方向余弦）
        print("tan_shape:", tan.shape)
        # print("tan:\n", tan)
        C, S = tan[:, 0], tan[:, 1]

        T = np.zeros((NC, GD*ldof, GD*ldof))
        
        T[:, 0, 0] = C
        T[:, 0, 1] = S
        T[:, 1, 0] = -S
        T[:, 1, 1] = C
        T[:, 2, 2] = 1
        T[:, 3, 3] = C
        T[:, 3, 4] = S
        T[:, 4, 3] = -S
        T[:, 4, 4] = C
        T[:, 5, 5] = 1

        print("单元的坐标变换矩阵 T_shape:", T.shape)
        # print("T:\n", T)

        K = np.einsum('nki, nkj, njl -> nil', T, K, T)

        if out is None:
            return K


