from fealpy.backend import backend_manager as bm

class DivergenceReconstruct:
    """
    Divergence reconstruction for finite volume method.
    This class provides methods to compute the divergence of a velocity field
    defined on edges of a mesh.
    """
    def __init__(self, mesh):
        self.mesh = mesh

    def StagReconstruct(self, edge_velocity):
        pem = self.mesh.entity_measure('edge')
        veloin = edge_velocity*pem
        PNC = self.mesh.number_of_cells()
        pe2c = self.mesh.edge_to_cell()[:,:2]
        bd_idx = self.mesh.boundary_face_index()
        div_u = bm.zeros(PNC)
        mask = pe2c[:, 1] != pe2c[:, 0]  # 非边界边
        bm.add.at(div_u, pe2c[mask, 0], veloin[mask])   # 左侧/下侧单元正贡献
        bm.add.at(div_u, pe2c[mask, 1], -veloin[mask])
        # bd_u = edge_velocity[bd_idx]
        # bd_n = self.mesh.edge_normal()[[bd_idx]]
        # bd_n = bm.sum(bd_n, axis=2, keepdims=True)
        # bd_in = bm.einsum('j,ijk->j', bd_u, bd_n)
        # bm.add.at(div_u, pe2c[bd_idx, 0], bd_in)
        return div_u
    
    def Reconstruct(self, edge_velocity):
        Sf = self.mesh.edge_normal()
        integrator = bm.einsum('ij,ij->i', edge_velocity, Sf)
        e2c = self.mesh.edge_to_cell()[:,:2]
        div_u = bm.zeros(self.mesh.number_of_cells())
        mask = e2c[:, 1] != e2c[:, 0]
        bm.add.at(div_u, e2c[:, 0], integrator)
        bm.add.at(div_u, e2c[mask, 1], -integrator[mask])
        return div_u
