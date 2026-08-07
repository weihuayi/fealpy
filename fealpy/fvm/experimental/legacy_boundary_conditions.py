"""Legacy boundary helpers used only by experimental FVM routes."""

from fealpy.backend import backend_manager as bm
from fealpy.sparse import spdiags

from ..dirichlet_bc import DirichletBC
from ..fvm_geometry import FVMGeometry
from ..neumann_bc import NeumannBC


class ExperimentalDirichletBC(DirichletBC):
    """Dirichlet boundary helper with legacy value-pinning operations."""

    def ThresholdApply(self, A, f, uh=None):
        total_bd_idx = self.mesh.boundary_cell_index()
        points = self.mesh.entity_barycenter('cell')
        NC = self.mesh.number_of_cells()
        bd_node = points[total_bd_idx]
        if callable(self.threshold):
            try:
                x = bd_node[:, 0]
                bd_idx = bm.array(self.threshold(x), dtype=bm.bool)
                if not bm.any(bd_idx):
                    y = bd_node[:, 1]
                    bd_idx = bm.array(self.threshold(y), dtype=bm.bool)
            except Exception:
                bd_idx = bm.array(self.threshold(bd_node), dtype=bm.bool)
        else:
            raise ValueError("self.threshold must be a callable boundary selector.")
        index = total_bd_idx[bd_idx]
        bdFlag_u = bm.zeros(NC, dtype=getattr(f, "dtype", None))
        bdFlag_u = bm.set_at(bdFlag_u, index, 1)
        D0 = spdiags(1 - bdFlag_u, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdFlag_u, 0, A.shape[0], A.shape[0])
        if uh is None:
            if hasattr(A, 'values_context'):
                uh = bm.zeros(A.shape[0], **A.values_context())
            else:
                uh = bm.zeros(A.shape[0], dtype=A.dtype)
        uh = bm.set_at(uh, index, self.gd(points[index]))
        f = f - A @ uh
        f = bm.set_at(f, index, uh[index])
        return D0.matmul(A.matmul(D0)) + D1, f

    def DivApply(self, b):
        NC = self.mesh.number_of_cells()
        geometry = self.geometry if self.geometry is not None else FVMGeometry(self.mesh)
        bd_edge = bm.nonzero(geometry.is_boundary)[0]
        bdedgepoint = geometry.face_center[bd_edge]
        bdSf = geometry.S_f[bd_edge]
        bde2c = geometry.owner[bd_edge]
        bd_u = self.gd(bdedgepoint)
        bd_correct = bd_u * bdSf
        bd_correct = bm.swapaxes(bd_correct, 0, 1).flatten()
        bd_correct = bm.array(bd_correct, dtype=b.dtype)
        bde2c = bm.concat([bde2c, bde2c + NC])
        return bm.index_add(b, bde2c, bd_correct, axis=0, alpha=-1)


class ExperimentalNeumannBC(NeumannBC):
    """Neumann helper with legacy RC boundary matrix operations."""

    def ThresholdApply(self, A, f, uh=None):
        total_bd_idx = self.mesh.boundary_cell_index()
        points = self.mesh.entity_barycenter('cell')
        NC = self.mesh.number_of_cells()
        bd_node = points[total_bd_idx]
        if callable(self.threshold):
            try:
                x = bd_node[:, 0]
                bd_idx = bm.array(self.threshold(x), dtype=bm.bool)
                if not bm.any(bd_idx):
                    y = bd_node[:, 1]
                    bd_idx = bm.array(self.threshold(y), dtype=bm.bool)
            except Exception:
                bd_idx = bm.array(self.threshold(bd_node), dtype=bm.bool)
        else:
            raise ValueError("self.threshold must be a callable boundary selector.")
        index = total_bd_idx[bd_idx]
        bdFlag_u = bm.zeros(NC, dtype=getattr(f, "dtype", None))
        bdFlag_u = bm.set_at(bdFlag_u, index, 1)
        D0 = spdiags(1 - bdFlag_u, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdFlag_u, 0, A.shape[0], A.shape[0])
        if uh is None:
            if hasattr(A, 'values_context'):
                uh = bm.zeros(A.shape[0], **A.values_context())
            else:
                uh = bm.zeros(A.shape[0], dtype=A.dtype)
        uh = bm.set_at(uh, index, self.gd(points[index]))
        f = f - A @ uh
        f = bm.set_at(f, index, uh[index])
        return D0.matmul(A.matmul(D0)) + D1, f

    def ConvectionApplyX(self, A, b):
        NC = self.mesh.number_of_cells()
        geometry = FVMGeometry(self.mesh)
        bdIdx = bm.zeros(NC, dtype=geometry.S_f.dtype)
        bdedge = bm.nonzero(geometry.is_boundary)[0]
        bdIdx = bm.index_add(
            bdIdx,
            geometry.owner[bdedge],
            geometry.S_f[bdedge, 0],
            axis=0,
        )
        return A + spdiags(bdIdx, 0, A.shape[0], A.shape[1])

    def ConvectionApplyY(self, A, b):
        NC = self.mesh.number_of_cells()
        geometry = FVMGeometry(self.mesh)
        bdIdx = bm.zeros(NC, dtype=geometry.S_f.dtype)
        bdedge = bm.nonzero(geometry.is_boundary)[0]
        bdIdx = bm.index_add(
            bdIdx,
            geometry.owner[bdedge],
            geometry.S_f[bdedge, 1],
            axis=0,
        )
        return A + spdiags(bdIdx, 0, A.shape[0], A.shape[1])
