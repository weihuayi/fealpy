import numpy as np

from .truss_structure_integrator import TrussStructureIntegrator


class EulerBernoulliBeamStructureIntegrator:
    def __init__(self, E, I, q = 3):
        """
        EulerBermoulliBeamStructureIntegrator 类的初始化

        参数:
        E -- 杨氏模量
        I -- 惯性矩
        q -- 积分公式的等级，默认值为3
        """
        self.E = E
        self.I = I
        self.q = q 

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

        assert  len(space) == GD

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        l = cellmeasure
        c = self.E * self.I

        NC = len(cellmeasure)
        ldof = 2 # 一个单元两个自由度, @TODO 高次元的情形？本科毕业论文
        if out is None:
            K = np.zeros((NC, GD*ldof, GD*ldof), dtype=np.float64)
        else:
            assert out.shape == (NC, GD*ldof, GD*ldof)
            K = out

        K_values = np.array([
            [12, 6, -12, 6],
            [6, 4, -6, 2],
            [-12, -6, 12, -6],
            [6, 2, -6, 4]
        ], dtype=np.float64)

        K_values[1, [1, 3]] *= l
        K_values[3, [1, 3]] *= l

        K_matrix = K_values[np.newaxis, :, :]

        K = (c / l**3)[:, np.newaxis, np.newaxis] * K_matrix

        if out is None:
            return K


class TimoshenkoBeamStructureIntegrator:
    def __init__(self, E, I, A, q = 3):
        """
        TimoshenkoBeamStructureIntegrator 类的初始化

        参数:
        E -- 杨氏模量
        I -- 惯性矩
        A -- 梁的横截面积
        q -- 积分公式的等级，默认值为3
        """
        self.E = E
        self.I = I
        self.A = A
        self.q = q 

        self.truss_integrator = TrussStructureIntegrator(E, A, q)
        self.euler_bernoulli_integrator = EulerBernoulliBeamStructureIntegrator(E, I, q)

    def assembly_cell_matrix(self, space, index = np.s_[:],
                             cellmeasure = None, out = None):
        """
        组装单元网格的刚度矩阵

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

        assert  len(space) == GD

        if cellmeasure is None:
            cellmeasure = mesh.entity_measure('cell', index=index)

        # NC = len(cellmeasure)

        # K_truss = self.truss_integrator.assembly_cell_matrix(space, index, cellmeasure, out)
        # K_euler_bernoulli = self.euler_bernoulli_integrator.assembly_cell_matrix(space, index, cellmeasure, out)
        # print("K_truss:\n", K_truss)
        # print("K_euler_bernoulli:\n", K_euler_bernoulli)

        tan = mesh.cell_unit_tangent(index=index) # 计算单元的单位切向矢量（即轴线方向余弦）
        print("tan:\n", tan)
        alpha = np.arctan2(tan[0][1], tan[0][0])
        print("alpha:", alpha)
        T = np.zeros((6, 6))
        
        # Fill the transformation matrix based on the computed angle alpha
        T[0, :2] = [np.cos(alpha), np.sin(alpha)]
        T[1, :2] = [-np.sin(alpha), np.cos(alpha)]
        T[2, 2] = 1
        T[3, 3:5] = [np.cos(alpha), np.sin(alpha)]
        T[4, 3:5] = [-np.sin(alpha), np.cos(alpha)]
        T[5, 5] = 1
        print("T:", T)
