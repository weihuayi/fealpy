from ..backend import backend_manager as bm
from .vector_decomposition import VectorDecomposition
from ..sparse import spdiags

class DirichletBC():
    def __init__(self, mesh, gd):
        self.mesh = mesh
        self.gd = gd
    
    def apply(self,A,b):
        _, d = VectorDecomposition(self.mesh).centroid_vector_calculation()
        Ef_abs = VectorDecomposition(self.mesh).Sor()
        integrator = Ef_abs/ d  
        boundary_edge = self.mesh.boundary_face_index()
        edge_to_cell = self.mesh.edge_to_cell()
        bde2c = edge_to_cell[boundary_edge,0] 
        NE = self.mesh.number_of_edges()
        bd1 = bm.zeros((NE,))  
        bd1[boundary_edge] = integrator[boundary_edge]
        bdIdx = bm.zeros((A.shape[0]))  
        boundary_integrator = integrator[boundary_edge].reshape(-1)
        bm.add_at(bdIdx, bde2c, boundary_integrator)
        A_0 = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        A = A + A_0
        edge_middle_point = self.mesh.entity_barycenter('edge')  
        bdedgepoint = edge_middle_point[boundary_edge]
        boundary_integrator = integrator[boundary_edge]
        boundary_u = self.gd(bdedgepoint) 
        bm.add_at(b, bde2c, boundary_integrator*boundary_u)
        return A,b