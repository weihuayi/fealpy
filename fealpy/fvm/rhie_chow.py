from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.fvm import VectorDecomposition

class RhieChowInterpolation:
    """
    Rhie-Chow interpolation to prevent pressure-velocity decoupling
    in collocated grids for incompressible flow simulations.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.NC = mesh.number_of_cells()
        self.NF = mesh.number_of_faces()
        self.cell_centers = mesh.entity_barycenter("cell")
        self.cell_measures = mesh.entity_measure("cell")
        self.edge_to_cell = mesh.edge_to_cell()
        self.cell_to_edge = mesh.cell_to_edge()

    def Ucell2edge(self,u):
        u = bm.stack([u[:self.NC],u[self.NC:]],axis=-1)
        bd_edge = self.mesh.boundary_face_index()
        edge_middle_point = self.mesh.entity_barycenter('edge')
        e2c = self.mesh.edge_to_cell()
        bdedgepoint = edge_middle_point[bd_edge]
        # bdedgeu = gd(bdedgepoint)
        uf = bm.zeros((self.mesh.number_of_faces(), 2), dtype=u.dtype)
        uf += (u[e2c[:,0]]+u[e2c[:,1]])/2
        # uf[bd_edge, :] = bdedgeu
        uf = bm.reshape(uf, (-1, 2))
        return uf
    
    def interpolate(self, u, p, mu):
        uf = self.Ucell2edge(u)
        e, d = VectorDecomposition(self.mesh).centroid_vector_calculation()
        partial_p = (p[self.edge_to_cell[:,1]] - p[self.edge_to_cell[:,0]])/d
