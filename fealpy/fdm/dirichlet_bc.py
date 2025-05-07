
from ..backend import backend_manager as bm
from ..typing import TensorLike
from ..sparse import spdiags

class DirichletBC():

    def __init__(self, mesh, gd, threshold=None):
        self.mesh = mesh
        self.gd = gd
        self.threshold = threshold

    def apply(self, A, f, uh=None):
        """
        """
        if uh is None:
            uh = bm.zeros(A.shape[0], **A.values_context())

        node = self.mesh.entity('node')
        bdFlag = self.mesh.boundary_node_flag()
        uh = bm.set_at(uh, bdFlag, self.gd(node[bdFlag]))
        
        f = f - A @ uh 
        f = bm.set_at(f, bdFlag, uh[bdFlag])

        D0 = spdiags(1-bdFlag, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdFlag, 0, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1
        return A, f
