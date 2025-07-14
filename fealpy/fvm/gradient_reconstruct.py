from fealpy.backend import backend_manager as bm
from .vector_decomposition import VectorDecomposition

class GradientReconstruct:
    def __init__(self, mesh):
        self.mesh = mesh

    def old_reconstruct(self, uh):
        Sf = VectorDecomposition(self.mesh).outer_normal_vector_calculation()
        cell2cell = self.mesh.cell_to_cell()
        S = self.mesh.entity_measure("cell")
        uh_nb = uh[cell2cell]           
        uh_f = 0.5 * (uh[:, None] + uh_nb) 
        grad = bm.sum(uh_f[:, :, None] * Sf, axis=1) / S[:, None]
        grad_nb = grad[cell2cell]  # (NC, 3, 2)
        grad_f = 0.5 * (grad[:, None, :] + grad_nb)  # (NC, 3, 2)
        return grad_f 

    def reconstruct(self, uh):
        Sf = self.mesh.edge_normal()
        e2c = self.mesh.edge_to_cell()  # (NE, 3)，其中 e2c[:, 0], e2c[:, 1] 是邻接单元
        cell_measure = self.mesh.entity_measure('cell')  # (NC,)
        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()
        GD = 2

        # --- Step 1: 计算边上的数值解 uh_f = (u_i + u_j)/2 ---
        uh_i = uh[e2c[:, 0]]
        uh_j = uh[e2c[:, 1]]
        uh_f = 0.5 * (uh_i + uh_j)  # (NE,)
        grad_u = bm.zeros((NC, GD))  # (NC, 2)
        for i in range(NE):
            c0 = e2c[i, 0]
            c1 = e2c[i, 1]
            Sf_i = Sf[i]
            grad_u[c0] += uh_f[i] * Sf_i
            # 边界边只出现一次，需要跳过第二次添加
            if c0 != c1:
                grad_u[c1] -= uh_f[i] * Sf_i
        grad_u /= cell_measure[..., None]  # (NC, 2)，每个单元的梯度
        # --- Step 4: 插值得到边上的重构梯度 ---
        grad_i = grad_u[e2c[:, 0]]  # (NE, 2)
        grad_j = grad_u[e2c[:, 1]]  # (NE, 2)
        grad_f = 0.5 * (grad_i + grad_j)  # (NE, 2)

        return grad_f  # 每条边上重构出来的梯度，方向与矢量一致