from fealpy.backend import backend_manager as bm
from ..fvm import ScalarDiffusionIntegrator
from .vector_decomposition import VectorDecomposition
from fealpy.sparse import SparseTensor, COOTensor, CSRTensor, spdiags

class DirichletBC():
    def __init__(self, mesh, gd):
        self.mesh = mesh
        self.gd = gd
    
    def apply(self,A,f):
        _, d = VectorDecomposition(self.mesh).centroid_vector_calculation()
        Ef_abs = VectorDecomposition(self.mesh).Sor()
        integrator = Ef_abs/ d  
        boundary_edge = self.mesh.boundary_face_index()
        edge_to_cell = self.mesh.edge_to_cell()
        bde2c = edge_to_cell[boundary_edge,0] 
        NE = self.mesh.number_of_edges()
        bd1 = bm.zeros((NE, 1))  
        bd1[boundary_edge] = integrator[boundary_edge]
        bdIdx = bm.zeros((A.shape[0]))  # 用于填充边界边对单元的贡献
        boundary_integrator = integrator[boundary_edge].reshape(-1)
        # for i in range(len(boundary_edge)):
        #     bdIdx[bde2c[i]] += boundary_integrator[i]
        bm.add_at(bdIdx, bde2c, boundary_integrator)  # Add boundary integrator to bdIdx
        A_0 = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        A = A + A_0
        edge_middle_point = self.mesh.bc_to_point(bm.array([0.5, 0.5]))
        bdedgepoint = edge_middle_point[boundary_edge]
        boundary_u = self.gd(bdedgepoint)
        bm.add_at(f, bde2c, boundary_u) 
        return A,f
    

    def apply_cross():

        pass