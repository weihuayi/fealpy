from fealpy.backend import backend_manager as bm
from ..fvm import ScalarDiffusionIntegrator

class DirichletBC:
    def __init__(self, mesh, gd):
        self.mesh = mesh
        self.gd = gd
        self.integrator = ScalarDiffusionIntegrator(mesh)
    
    def apply(self,f):
        Ef,_ = self.integrator.centroid_vector_calculation()
        if self.mesh.GD == 2:
            cell_centers = self.mesh.bc_to_point(bm.array([1/3, 1/3, 1/3]))
            edge_middle_point = self.mesh.bc_to_point(bm.array([0.5, 0.5]))
        elif self.mesh.GD == 3:
            cell_centers = self.mesh.bc_to_point(bm.array([1/4, 1/4, 1/4, 1/4]))
            edge_middle_point = self.mesh.bc_to_point(bm.array([1/3, 1/3, 1/3]))
        e2c = self.mesh.edge_to_cell()
        boundary_edge = self.mesh.boundary_face_index()
        boundary_e2c = e2c[boundary_edge]
        boundary_meshcenter = cell_centers[boundary_e2c[..., 0]]
        boundary_edge_middle_point = edge_middle_point[boundary_edge]        
        e = boundary_edge_middle_point - boundary_meshcenter
        d = bm.linalg.norm(e, axis=-1, keepdims=True)
        boundary_u = self.gd(boundary_edge_middle_point)[..., None]
        boundary_ef = Ef[boundary_e2c[..., 0], boundary_e2c[..., 2]]
        boundary_ef_norm = bm.linalg.norm(boundary_ef, axis=-1)[..., None]
        boundary_flux = (boundary_ef_norm / d)*boundary_u
        f[boundary_e2c[..., 0]] += boundary_flux
        return f