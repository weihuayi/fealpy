from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike
from fealpy.fvm import VectorDecomposition,GradientReconstruct
from fealpy.model import PDEModelManager
class RhieChowInterpolation:
    """
    Rhie-Chow interpolation to prevent pressure-velocity decoupling
    in collocated grids for incompressible flow simulations.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        # self.gd = gd
        self.cm = self.mesh.entity_measure('cell')
        self.NC = mesh.number_of_cells()
        self.NF = mesh.number_of_faces()
        self.cell_centers = mesh.entity_barycenter("cell")
        self.cell_measures = mesh.entity_measure("cell")
        self.edge_to_cell = mesh.edge_to_cell()
        self.cell_to_edge = mesh.cell_to_edge()

    def Ucell2edge(self,u,ap):
        
        ap = ap[:self.NC][:,None]
        dp = self.cm[:,None]/ap
        u = bm.stack([u[:self.NC],u[self.NC:]],axis=-1)
        # bd_edge = self.mesh.boundary_face_index()
        # edge_middle_point = self.mesh.entity_barycenter('edge')
        e2c = self.mesh.edge_to_cell()
        # bdedgepoint = edge_middle_point[bd_edge]
        # bdedgeu = self.gd(bdedgepoint)
        uf = bm.zeros((self.mesh.number_of_faces(), 2), dtype=u.dtype)
        df = bm.zeros(self.mesh.number_of_faces())
        # x = ap[e2c[:,0]]*u[e2c[:,0]]+ap[e2c[:,1]]*u[e2c[:,1]]
        # y = ap[e2c[:,0]]+ap[e2c[:,1]]
        # uf = x/y
        uf = (u[e2c[:,0]]+u[e2c[:,1]])/2
        df = (dp[e2c[:,1]]+dp[e2c[:,0]])/2
        # uf[bd_edge, :] = bdedgeu
        return uf,df
    
    def GradientDifference(self, p):
        """
        Gradient difference calculation
        """
        e, d = VectorDecomposition(self.mesh).centroid_vector_calculation()
        partial_p = (p[self.edge_to_cell[:,1]] - p[self.edge_to_cell[:,0]])/d
        e_cf = e / d[:, None]
        pde = PDEModelManager("navier_stokes").get_example(1)
        grad_p = GradientReconstruct(
            self.mesh).AverageGradientreNeumann(
                p,gd = pde.neumann_pressure)
        
        overline_grad_p_f = GradientReconstruct(self.mesh).reconstruct(grad_p)
        GradientDifference = (partial_p - bm.einsum('ij,ij->i', overline_grad_p_f, e_cf))[:, None]*e_cf
        # print("GradientDifference:",GradientDifference)
        # x = GradientDifference.shape[0]
        # print("l2GradientDifference:",GradientDifference[int(x/2)])
        return GradientDifference
    
    def Interpolation(self,u,ap,p):
        """
        Perform Rhie-Chow interpolation
        """
        uf, df = self.Ucell2edge(u,ap)
        grad_diff = self.GradientDifference(p)
        return uf - df*grad_diff
