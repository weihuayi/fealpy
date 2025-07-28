from fealpy.backend import backend_manager as bm
from fealpy.sparse import spdiags

from .vector_decomposition import VectorDecomposition

class DirichletBC():
    def __init__(self, mesh, gd):
        self.mesh = mesh
        self.gd = gd
    
    def DiffusionApply(self,A,b):
        bd_edge = self.mesh.boundary_face_index()
        e2c = self.mesh.edge_to_cell()
        NC = self.mesh.number_of_cells()
        _, d = VectorDecomposition(self.mesh).centroid_vector_calculation()
        Ef_abs = VectorDecomposition(self.mesh).Sor()
        bd_integrator = Ef_abs[bd_edge]/ d[bd_edge]
        bde2c = e2c[bd_edge,0] 
        edge_middle_point = self.mesh.entity_barycenter('edge')  
        bdedgepoint = edge_middle_point[bd_edge]
        #标量场下bd_u形状为(NE,),二维矢量场下为(NE, 2),三维矢量场下为(NE, 3)，为了判断维度给bd_u加一个轴
        bd_u = self.gd(bdedgepoint)[..., None] 
          
        bdIdx = bm.zeros(NC)  
        bm.add_at(bdIdx, bde2c, bd_integrator)
        #通过bdIdx的第二个轴来判断是标量场，二维矢量场还是三维矢量场   
        D = bd_u.shape[1]
        bdIdx = bm.tile(bdIdx,D)
        A_0 = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        A = A + A_0
        if D == 1:
            bd_correct = (bd_integrator[:, None]*bd_u).reshape(-1)
            bm.add_at(b, bde2c, bd_correct) 
        else:
            #需要去掉bd_u增加的那个轴以便于计算
            bd_u = bm.squeeze(bd_u, axis=-1)
            bd_correct = bd_integrator[:, None]*bd_u
            bd_correct = bm.transpose(bd_correct).flatten()
            new_arr = bde2c + NC
            bde2c = bm.concat([bde2c, new_arr])
            bm.add_at(b, bde2c, bd_correct) 
        return A,b
    
    def DivApply(self,b):
        NC = self.mesh.number_of_cells()
        n = self.mesh.face_unit_normal()
        facemeasure = self.mesh.entity_measure('face')
        bd_edge = self.mesh.boundary_face_index()
        edge_middle_point = self.mesh.entity_barycenter('edge') 
        e2c = self.mesh.edge_to_cell()
        bdedgepoint = edge_middle_point[bd_edge]
        bdSf = (facemeasure[:, None] * n)[bd_edge]  # (bdNE, 2)
        bde2c = e2c[bd_edge,0] 
        #二维矢量场下bd_u为(bdNE, 2),三维矢量场下为(bdNE, 3)
        bd_u = self.gd(bdedgepoint)
        bd_correct = bd_u*bdSf
        bd_correct = bm.transpose(bd_correct).flatten()
        new_arr = bde2c + NC
        bde2c = bm.concat([bde2c, new_arr])        
        bm.add_at(b, bde2c, -bd_correct) 
        return b