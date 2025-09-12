from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

class VectorDecomposition():
    def __init__(self, mesh):
        self.mesh = mesh  
        self.Sf = mesh.edge_normal()  # (NE, 2)

    def centroid_vector_calculation(self) -> TensorLike:
        cell_centers = self.mesh.entity_barycenter('cell')  # (NC, 2)
        edge_middle_point = self.mesh.entity_barycenter('face')
        e2c = self.mesh.edge_to_cell()
        NE = self.mesh.number_of_edges()
        e = bm.zeros((NE, 2))
        is_interior = e2c[:, 0] != e2c[:, 1]
        e[is_interior] = cell_centers[e2c[:, 1][is_interior]] - cell_centers[e2c[:, 0][is_interior]]
        is_boundary = ~is_interior 
        e[is_boundary] = edge_middle_point[is_boundary] - cell_centers[e2c[:, 0][is_boundary]]
        d = bm.linalg.norm(e, axis=-1, keepdims=True).reshape(-1)
        return e, d
    
    def Sor(self) -> TensorLike:
        e, _ = self.centroid_vector_calculation()
        Sf_dot_Sf = bm.einsum('ij,ij->i', self.Sf, self.Sf).reshape(-1, 1)  
        e_dot_Sf = bm.einsum('ij,ij->i', e, self.Sf).reshape(-1, 1)    
        e_norm = bm.linalg.norm(e, axis=-1, keepdims=True)  
        Ef_abs = (Sf_dot_Sf / e_dot_Sf) * e_norm
        return Ef_abs.reshape(-1)   # shape: (NE, 1)

    def tangential_vector_calculation(self) -> TensorLike:
        e, _ = self.centroid_vector_calculation()    # (NE, 2)
        dot_e_Sf = bm.sum(e * self.Sf, axis=1, keepdims=True)  # (NE, 1)
        norm_Sf_sq = bm.sum(self.Sf * self.Sf, axis=1, keepdims=True)
        Ef = (norm_Sf_sq / (dot_e_Sf + 1e-13)) * e
        Tf = self.Sf - Ef
        return Tf  # shape: (NE, 2)
    