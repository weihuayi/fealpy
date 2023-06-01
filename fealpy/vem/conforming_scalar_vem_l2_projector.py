import numpy as np 
from numpy.linalg import inv

from fealpy.functionspace import ConformingScalarVESpace2d
from fealpy.vem.conforming_scalar_vem_h1_projector import ConformingScalarVEMH1Projector2d
class ConformingScalarVEML2Projector2d():
    def __init__(self):
        pass

    def assembly_cell_matrix(self, space: ConformingScalarVESpace2d):
        H = space.smspace.matrix_H()
        C = self.assembly_cell_righthand_side(space) 
        pi0 = lambda x: inv(x[0])@x[1]
        return list(map(pi0, zip(H, C)))


    def assembly_cell_righthand_side(self, space: ConformingScalarVESpace2d):
        """
        @brief 组装 L2 投影算子的右端矩阵

        @retrun C 列表 C[i] 代表第 i 个单元上 L2 投影右端矩阵
        """
        p = space.p
        mesh = space.mesh
        NV = mesh.ds.number_of_vertices_of_cells()

        idof = (p-1)*p//2
        smldof = space.smspace.number_of_local_dofs()
        H = space.smspace.matrix_H()
        H1Projector = ConformingScalarVEMH1Projector2d()
        PI1 = H1Projector.assembly_cell_H1_matrix(space)

        d = lambda x: x[0]@x[1]
        C = list(map(d, zip(H, PI1)))
        if p == 1:
            return C
        else:
            l = lambda x: np.r_[
                    '0',
                    np.r_['1', np.zeros((idof, p*x[0])), x[1]*np.eye(idof)],
                    x[2][idof:, :]]
            return list(map(l, zip(NV, space.smspace.cellmeasure, C)))


