from fealpy.backend import backend_manager as bm
from scipy.sparse import coo_matrix

class BilinearForm():
    def __init__(self, mesh, pde):
        self.mesh = mesh
        self.pde = pde

    def matrix_assembly(self, flux_coeff):
        cell2cell = self.mesh.cell_to_cell()
        NC = self.mesh.number_of_cells()
        LNF = self.mesh.number_of_faces_of_cells()
        A = bm.zeros((NC, NC))
        I = []  
        J = []  
        V = []  
        for i in range(LNF):
            src = bm.arange(NC)
            tgt = cell2cell[:, i]
            cval = -flux_coeff[:, i]
            mask = (tgt != src)  # 非对角项（处理非边界或边界有邻居的）
            I.extend(src[mask])
            J.extend(tgt[mask])
            V.extend(cval[mask])
        diag = bm.sum(flux_coeff, axis=1)
        I.extend(bm.arange(NC))
        J.extend(bm.arange(NC))
        V.extend(diag)
        A = coo_matrix((V, (I, J)), shape=(NC, NC))
        return A