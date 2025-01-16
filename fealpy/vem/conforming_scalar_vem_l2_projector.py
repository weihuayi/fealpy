import numpy as np 
from numpy.linalg import inv

from ..functionspace import ConformingScalarVESpace2d

class ConformingScalarVEML2Projector2d():
    def __init__(self, M, PI1):
        self.M = M
        self.PI1 = PI1

    def assembly_cell_matrix(self, space: ConformingScalarVESpace2d):
        self.C = self.assembly_cell_right_hand_side(space)
        pi0 = lambda x: inv(x[0])@x[1]
        return list(map(pi0, zip(self.M, self.C))) # TODO：并行加速

    def assembly_cell_right_hand_side(self, space: ConformingScalarVESpace2d):
        """
        @brief 组装 L2 投影算子的右端矩阵

        @retrun C 列表 C[i] 代表第 i 个单元上 L2 投影右端矩阵
        """
        p = space.p
        mesh = space.mesh
        NV = mesh.ds.number_of_vertices_of_cells()

        idof = (p-1)*p//2
        smldof = space.smspace.number_of_local_dofs()

        d = lambda x: x[0]@x[1]
        C = list(map(d, zip(self.M, self.PI1))) #TODO: 并行加速
        if p == 1:
            return C
        else:
            l = lambda x: np.r_[
                    '0',
                    np.r_['1', np.zeros((idof, p*x[0])), x[1]*np.eye(idof)],
                    x[2][idof:, :]]
            return list(map(l, zip(NV, space.smspace.cellmeasure, C))) #TODO：并行加速


