import numpy as np

class BeamStructureIntegrator:
    def __init__(self, E, A, I, G, q = 3):
        """
        BeamStructureIntegrator 类的初始化

        参数:
        E -- 杨氏模量
        A -- 单元横街面积
        I -- 惯性矩
        G -- 剪切模量
        q -- 积分公式的等级，默认值为3
        """
        self.E = E
        self.A = A
        self.I = I
        self.G = G
        self.q = q 

            
    def assembly_cell_matrix(self, space, index, cellmeasure, out):
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

        NC = len(cellmeasure)

        c = self.E*self.I
        tan = mesh.cell_unit_tangent(index=index) # 计算单元的单位切向矢量（即轴线方向余弦）
        print("tan_shape:", tan.shape)
        print("tan:\n", tan)

        R = np.einsum('ik, im->ikm', tan, tan)
        print("R_shape:", R.shape)
        print("R:\n", R)
        R *= c/cellmeasure[:, None, None]
        print("R:\n", R)

        ldof = 2 # 一个单元两个自由度, @TODO 高次元的情形？本科毕业论文
        if out is None:
            K = np.zeros((NC, GD*ldof, GD*ldof), dtype=np.float64)
        else:
            assert out.shape == (NC, GD*ldof, GD*ldof)
            K = out
