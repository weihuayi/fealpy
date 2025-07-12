from fealpy.backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike

class VectorDecomposition():
    def __init__(self, mesh):
        self.mesh = mesh  


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
        d = bm.linalg.norm(e, axis=-1, keepdims=True)
        return e, d
    
    def Sor(self):
        n = self.mesh.face_unit_normal()
        e, _ = self.centroid_vector_calculation()
        facemeasure = self.mesh.entity_measure('face')
        Sf = facemeasure[:, None] * n  # (NE, 2)
        Sf_dot_Sf = bm.einsum('ij,ij->i', Sf, Sf).reshape(-1, 1)  
        e_dot_Sf = bm.einsum('ij,ij->i', e, Sf).reshape(-1, 1)    
        e_norm = bm.linalg.norm(e, axis=-1, keepdims=True)  
        Ef_abs = (Sf_dot_Sf / e_dot_Sf) * e_norm 
        return Ef_abs  # shape: (NE, 1)

    def tangential_vector_calculation(self):
        n = self.mesh.face_unit_normal()
        facemeasure = self.mesh.entity_measure('face')
        Sf = facemeasure[:, None] * n  # (NE, 2)
        e, _ = self.centroid_vector_calculation()    # (NE, 2)
        dot_e_Sf = bm.sum(e * Sf, axis=1, keepdims=True)  # (NE, 1)
        norm_Sf_sq = bm.sum(Sf * Sf, axis=1, keepdims=True)
        Ef = (norm_Sf_sq / (dot_e_Sf + 1e-13)) * e
        Tf = Sf - Ef
        return Tf  # shape: (NE, 2)
    
    def outer_normal_vector_calculation(self):
        cell = self.mesh.entity("cell")
        node = self.mesh.entity("node")
        p = node[cell]
        if self.mesh.GD == 2:
            edge_vec = bm.stack([p[:, 2] - p[:, 1], 
                            p[:, 0] - p[:, 2], 
                            p[:, 1] - p[:, 0]], axis=1)
            x = bm.array([[0, -1], [1, 0]])
            edge_vec_unit = edge_vec / bm.linalg.norm(edge_vec, axis=-1, keepdims=True)
            en = edge_vec_unit @ x
            edge_vec_norm = bm.linalg.norm(edge_vec, axis=-1, keepdims=True)
            Sf = bm.einsum('ijl,ijk->ijk', edge_vec_norm, en)
        elif self.mesh.GD == 3:
            local_faces = bm.array([
                [1, 2, 3],  # face 0
                [0, 3, 2],  # face 1
                [0, 1, 3],  # face 2
                [0, 2, 1],  # face 3
            ])
            face_coords = p[:, local_faces]
            vec1 = face_coords[:, :, 1] - face_coords[:, :, 0]  # (NC, 4, 3)
            vec2 = face_coords[:, :, 2] - face_coords[:, :, 0]  # (NC, 4, 3)
            Sf = 0.5 * bm.cross(vec1, vec2) # (NC, 4, 3)
        return Sf
    
    def old_tangential_vector_calculation(self):
        cell2cell = self.mesh.cell_to_cell()
        NC = self.mesh.number_of_cells()
        LNF = self.mesh.number_of_faces_of_cells()
        GD = self.mesh.GD
        Sf = self.outer_normal_vector_calculation()
        if GD == 2:
            cell_centers = self.mesh.bc_to_point(bm.array([1/3, 1/3, 1/3]))
            edge_middle_point = self.mesh.bc_to_point(bm.array([0.5, 0.5]))
        elif GD == 3:
            cell_centers = self.mesh.bc_to_point(bm.array([1/4, 1/4, 1/4, 1/4]))
            edge_middle_point = self.mesh.bc_to_point(bm.array([1/3, 1/3, 1/3]))
        e = bm.zeros((NC, LNF, GD))
        for j in range(LNF):
            nbr_ids = cell2cell[:, j]
            e[:, j, :] = cell_centers[nbr_ids] - cell_centers
        e2c = self.mesh.edge_to_cell()
        boundary_edge = self.mesh.boundary_face_index()
        boundary_e2c = e2c[boundary_edge]
        boundary_meshcenter = cell_centers[boundary_e2c[..., 0]]
        boundary_edge_middle_point = edge_middle_point[boundary_edge]        
        e2 = boundary_edge_middle_point - boundary_meshcenter
        cell_idx = boundary_e2c[:, 0]
        local_edge_idx = boundary_e2c[:, 2]
        e[cell_idx, local_edge_idx] = e2
        d = bm.linalg.norm(e, axis=-1, keepdims=True)
        numerator = bm.einsum('ijk,ijk->ij', Sf, Sf)[..., None]
        denominator = bm.einsum('ijk,ijk->ij', e, Sf)[..., None]
        Ef = (numerator / denominator) * e
        Tf = Sf - Ef
        return Tf
    