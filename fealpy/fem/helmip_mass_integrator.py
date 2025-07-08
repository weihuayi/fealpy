from typing import Optional
from ..backend import backend_manager as bm
from ..typing import TensorLike, Threshold, CoefLike
from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral
from .integrator import (
    LinearInt, OpInt, FaceInt,
    enable_cache
)

class HelmipMassIntegrator(LinearInt, OpInt, FaceInt):
    """
    A class for computing scalar face interior penalty integrals in finite element methods.
    
    Parameters:
        coef: Coefficient function or value for the integrator.
        q: Quadrature order for numerical integration. If None, defaults to p+3.
        threshold: Threshold for selecting faces (can be a callable or tensor).
        batched: Flag indicating whether to use batched computation.
    """
    def __init__(self, coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 threshold: Optional[Threshold]=None,
                 batched: bool=False):
        super().__init__()
        self.coef = coef
        self.q = q
        self.threshold = threshold
        self.batched = batched

    @enable_cache
    def make_index(self, space: _FS):
        """
        Create an index array identifying interior faces based on the threshold.

        Parameters:
            space: Function space object.

        Returns:
            index array.
        """
        threshold = self.threshold

        if isinstance(threshold, TensorLike):
            index = threshold
        else:
            mesh = space.mesh
            face2cell = mesh.face_to_cell()
            index = face2cell[:, 0] != face2cell[:, 1]
            if callable(threshold):
                bc = mesh.entity_barycenter('face', index=index)
                index = index[threshold(bc)]
        return index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        """
        Map local degrees of freedom to global degrees of freedom for faces.

        Parameters:
            space: Function space object.

        Returns:
            TensorLike: Array mapping face dofs to global dofs for faces.
        """
        index = self.make_index(space)
        mesh = getattr(space, 'mesh', None)
        p = getattr(space, 'p', None)

        TD = mesh.top_dimension()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()

        isFaceDof = (mesh.multi_index_matrix(p,TD) == 0)  # 判断多重指标哪些位置为 0
        cell2face = mesh.cell_to_face()
        cell2facesign = mesh.cell_to_face_sign()

        ldof = space.number_of_local_dofs() 
        fdof = space.number_of_local_dofs('face') 
        ndof = ldof - fdof
        face2dof = bm.zeros((NF, fdof + 2*ndof), dtype=bm.int32)   # *2: 左右两边
        cell2dof = space.cell_to_dof()

        for i in range(TD+1): 

            lidx, = bm.nonzero( cell2facesign[:, i])  # 单元是全局面的左边单元
            ridx, = bm.nonzero(~cell2facesign[:, i])  # 单元是全局面的右边单元
            idx0, = bm.nonzero( isFaceDof[:, i])  # 在面上的自由度
            idx1, = bm.nonzero(~isFaceDof[:, i])  # 不在面上的自由度
 
            fidx = cell2face[:, i]   # 第 i 个面的全局编号
            face2dof[fidx[lidx, None], bm.arange(fdof,      fdof+  ndof)] = cell2dof[lidx[:, None], idx1] 
            face2dof[fidx[ridx, None], bm.arange(fdof+ndof, fdof+2*ndof)] = cell2dof[ridx[:, None], idx1]

            # 面上的自由度按编号大小进行排序
            idx = bm.argsort(cell2dof[:, isFaceDof[:, i]], axis=1) 
            face2dof[fidx, 0:fdof] = cell2dof[:, isFaceDof[:, i]][bm.arange(NC)[:, None], idx] 

        return face2dof[index]

    @enable_cache
    def fetch(self, space: _FS):
        """
        Prepare quadrature points, weights, gradient of basis functions, measures for face, index.

        Parameters:
            space: Function space object.

        Returns:
            bcs, ws, phi, cm, index.
        """
        q = self.q
        index = self.make_index(space)
        mesh = getattr(space, 'mesh', None)
        p = getattr(space, 'p', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('face', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        NF = mesh.number_of_faces()
        TD = mesh.top_dimension()
        NQ = len(ws)

        ldof = space.number_of_local_dofs() 
        fdof = space.number_of_local_dofs('face') 
        ndof = ldof - fdof
        cell2face = mesh.cell_to_face()
        cell2dof = space.cell_to_dof()
        isFaceDof = (mesh.multi_index_matrix(p,TD) == 0) 
        cell2facesign = mesh.cell_to_face_sign()

        n = mesh.face_unit_normal()

        phi = bm.zeros((NF, NQ, fdof + 2*ndof),dtype=bm.float64)
        for i in range(TD+1):

            lidx, = bm.nonzero( cell2facesign[:, i]) 
            ridx, = bm.nonzero(~cell2facesign[:, i]) 
            idx0, = bm.nonzero( isFaceDof[:, i]) 
            idx1, = bm.nonzero(~isFaceDof[:, i]) 

            fidx = cell2face[:, i] 
            idx = bm.argsort(cell2dof[:, isFaceDof[:, i]], axis=1) 

            # 面上的积分点转化为体上的积分点
            b = bm.insert(bcs, i, 0, axis=1)

            cval = bm.einsum('cqlm, cm->cql', space.grad_basis(b), n[cell2face[:, i]])
            phi[fidx[ridx, None],:, bm.arange(fdof+ndof, fdof+2*ndof)] = +cval[ridx[:, None],:, idx1]
            phi[fidx[lidx, None],:, bm.arange(fdof,      fdof+  ndof)] = -cval[lidx[:, None],:, idx1]

            phi[fidx[ridx, None],:, bm.arange(0, fdof)] += cval[ridx[:, None],:, idx0[idx[ridx, :]]]
            phi[fidx[lidx, None],:, bm.arange(0, fdof)] -= cval[lidx[:, None],:, idx0[idx[lidx, :]]] 
        phi = phi[index]

        return bcs, ws, phi, cm, index

    def assembly(self, space: _FS) -> TensorLike:
        '''Assemble the matrix for interior face penalty terms.'''
        coef = self.coef
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, cm, index = self.fetch(space)
        val = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='face', index=index)

        return bilinear_integral(phi, phi, ws, cm*cm, val, batched=self.batched)