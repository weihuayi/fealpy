from fealpy.backend import backend_manager as bm
from ..fvm import GradientReconstruct

class ScalarDiffusionIntegrator():
    def __init__(self, mesh):
        self.mesh = mesh
        self.gradient_reconstruct = GradientReconstruct(mesh)     

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
    
    def centroid_vector_calculation(self):
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
        return Ef, d

    def vector_decomposition(self):
        Ef , d = self.centroid_vector_calculation()
        Sf = self.outer_normal_vector_calculation()
        LNF = self.mesh.number_of_faces_of_cells()
        NC = self.mesh.number_of_cells()
        Ef_norm = bm.linalg.norm(Ef, axis=-1)[..., None]
        Tf = Sf - Ef
        flux_coeff = Ef_norm / d
        flux_coeff = flux_coeff.reshape((NC, LNF))
        return flux_coeff, Tf

    def Cross_diffusion(self,uh, Tf):
        Sf = self.outer_normal_vector_calculation()
        grad_f = self.gradient_reconstruct.gradientreconstruct(uh, Sf) 
        Cross_diffusion = bm.einsum('ijk,ijk->i', Tf, grad_f)[..., None]
        return Cross_diffusion