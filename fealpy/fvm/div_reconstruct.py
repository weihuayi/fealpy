from fealpy.backend import backend_manager as bm

class DivergenceReconstruct:
    """
    Divergence reconstruction for finite volume method.
    This class provides methods to compute the divergence of a velocity field
    defined on edges of a mesh.
    """
    def __init__(self, mesh):
        self.mesh = mesh

    def reconstruct(self, edge_velocity, gd):
        pem = self.mesh.entity_measure('edge')
        veloin = edge_velocity*pem
        PNC = self.mesh.number_of_cells()
        pe2c = self.mesh.edge_to_cell()[:,:2]
        bd_idx = self.mesh.boundary_face_index()
        div_u = bm.zeros(PNC)
        mask = pe2c[:, 1] != pe2c[:, 0]  # 非边界边
        bm.add.at(div_u, pe2c[mask, 0], veloin[mask])   # 左侧/下侧单元正贡献
        bm.add.at(div_u, pe2c[mask, 1], -veloin[mask])
        epoints = self.mesh.entity_barycenter('edge')
        bd_u = gd(epoints[bd_idx])
        bd_n = self.mesh.edge_normal()[[bd_idx]]
        bd_in = bm.einsum('jk,ijk->j', bd_u, bd_n)
        bm.add.at(div_u, pe2c[bd_idx, 0], bd_in)
        return div_u

