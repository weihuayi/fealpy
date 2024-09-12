from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S

from ..mesh import HomogeneousMesh, SimplexMesh
from ..functionspace.space import FunctionSpace as _FS
from fealpy.experimental.functionspace.tensor_space import TensorFunctionSpace as _TS
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod
)

class LinearElasticIntegrator(LinearInt, OpInt, CellInt):
    """
    The linear elastic integrator for function spaces based on homogeneous meshes.
    """
    def __init__(self, 
                 material,
                 q: Optional[int]=None, *,
                 index: Index=_S,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)

        self.material = material
        self.q = q
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        index = self.index
        mesh = getattr(space, 'mesh', None)
    
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
    
        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index, variable='x')
        return bcs, ws, gphi, cm, index, q
    
    def assembly(self, space: _TS) -> TensorLike:
        scalar_space = space.scalar_space
        bcs, ws, gphi, cm, index, q = self.fetch(scalar_space)
        
        D = self.material.elastic_matrix(bcs)
        B = self.material.strain_matrix(dof_priority=space.dof_priority, gphi=gphi)

        KK = bm.einsum('q, c, cqki, cqkl, cqlj -> cij', ws, cm, B, D, B)
        
        return KK

    @assemblymethod('fast_strain')
    def fast_assembly_strain(self, space: _TS) -> TensorLike:
        index = self.index
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)

        if not isinstance(mesh, SimplexMesh):
            raise RuntimeError("The mesh should be an instance of SimplexMesh.")

        GD = mesh.geo_dimension()
        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()

        # (NQ, LDOF, BC)
        gphi_lambda = scalar_space.grad_basis(bcs, index=index, variable='u')
        # (LDOF, LDOF, BC, BC)
        M = bm.einsum('q, qik, qjl -> ijkl', ws, gphi_lambda, gphi_lambda)

        # (NC, LDOF, GD)
        glambda_x = mesh.grad_lambda()
        # (NC, LDOF, LDOF)
        A_xx = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 0], glambda_x[..., 0], cm)
        A_yy = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 1], glambda_x[..., 1], cm)
        A_xy = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 0], glambda_x[..., 1], cm)
        A_yx = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 1], glambda_x[..., 0], cm)

        NC = mesh.number_of_cells()
        ldof = scalar_space.number_of_local_dofs()
        tldof = space.number_of_local_dofs()
        KK = bm.zeros((NC, tldof, tldof), dtype=bm.float64)

        # TODO 只能处理 (NC, 1, 3, 3) 和 (1, 1, 3, 3) 的情况 
        D = self.material.elastic_matrix()
        if D.shape[0] != 1:
            raise ValueError("Elastic matrix D must have shape (NC, 1, 3, 3) or (1, 1, 3, 3).")
        D00 = D[..., 0, 0, None]
        D01 = D[..., 0, 1, None]
        D22 = D[..., 2, 2, None]
        
        if space.dof_priority:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof, 1), slice(0, ldof, 1)), D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1], 1), slice(ldof, KK.shape[1], 1)), D00 * A_yy + D22 * A_xx)
            # KK[:, 0:ldof:1, 0:ldof:1] = D00 * A_xx + D22 * A_yy
            # KK[:, ldof:KK.shape[1]:1, ldof:KK.shape[1]:1] = D00 * A_yy + D22 * A_xx

            # Fill the off-diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof, 1), slice(ldof, KK.shape[1], 1)), D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1], 1), slice(0, ldof, 1)), D01 * A_yx + D22 * A_xy)
            # KK[:, 0:ldof:1, ldof:KK.shape[1]:1] = D01 * A_xy + D22 * A_yx
            # KK[:, ldof:KK.shape[1]:1, 0:ldof:1] = D01 * A_yx + D22 * A_xy
        else:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), D00 * A_yy + D22 * A_xx)
            # KK[:, 0:KK.shape[1]:GD, 0:KK.shape[2]:GD] = D00 * A_xx + D22 * A_yy
            # KK[:, 1:KK.shape[1]:GD, 1:KK.shape[2]:GD] = D00 * A_yy + D22 * A_xx

            # Fill the off-diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), D01 * A_yx + D22 * A_xy)
            # KK[:, 0:KK.shape[1]:GD, 1:KK.shape[2]:GD] = D01 * A_xy + D22 * A_yx
            # KK[:, 1:KK.shape[1]:GD, 0:KK.shape[2]:GD] = D01 * A_yx + D22 * A_xy
        
        return KK

    @assemblymethod('fast_stress')
    def fast_assembly_stress(self, space: _TS) -> TensorLike:
        index = self.index
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)

        if not isinstance(mesh, SimplexMesh):
            raise RuntimeError("The mesh should be an instance of SimplexMesh.")
        
        GD = mesh.geo_dimension()
        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()

        # (NQ, LDOF, BC)
        gphi_lambda = scalar_space.grad_basis(bcs, index=index, variable='u')
        # (LDOF, LDOF, BC, BC)
        M = bm.einsum('q, qik, qjl->ijkl', ws, gphi_lambda, gphi_lambda)

        # (NC, LDOF, GD)
        glambda_x = mesh.grad_lambda()
        # (NC, LDOF, LDOF)
        A_xx = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 0], glambda_x[..., 0], cm)
        A_yy = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 1], glambda_x[..., 1], cm)
        A_xy = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 0], glambda_x[..., 1], cm)
        A_yx = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 1], glambda_x[..., 0], cm)

        NC = mesh.number_of_cells()
        ldof = scalar_space.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64)

        # TODO 只能处理 (NC, 1, 3, 3) 和 (1, 1, 3, 3) 的情况 
        D = self.material.elastic_matrix()
        if D.shape[1] != 1:
            raise ValueError("fast_assembly_stress currently only supports elastic matrices "
                            "with shape (NC, 1, 3, 3) or (1, 1, 3, 3).")
        D00 = D[..., 0, 0, None]
        D01 = D[..., 0, 1, None]
        D22 = D[..., 2, 2, None]

        if space.dof_priority:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(ldof, KK.shape[1])), D00 * A_yy + D22 * A_xx)
            # KK[:, 0:ldof, 0:ldof] = D00 * A_xx + D22 * A_yy
            # KK[:, ldof:KK.shape[1]:1, ldof:KK.shape[1]:1] = D00 * A_yy + D22 * A_xx

            # Fill the off-diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, KK.shape[1])), D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(0, ldof)), D22 * A_yx + D01 * A_xy)
            # KK[:, 0:ldof, ldof:KK.shape[1]:1] = D01 * A_xy + D22 * A_yx
            # KK[:, ldof:KK.shape[1]:1, 0:ldof] = D22 * A_yx + D01 * A_xy
        else:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), D00 * A_yy + D22 * A_xx)
            # KK[:, 0:KK.shape[1]:GD, 0:KK.shape[2]:GD] = D00 * A_xx + D22 * A_yy
            # KK[:, 1:KK.shape[1]:GD, 1:KK.shape[2]:GD] = D00 * A_yy + D22 * A_xx

            # Fill the off-diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), D22 * A_yx + D01 * A_xy)
            # KK[:, 0:KK.shape[1]:GD, 1:KK.shape[2]:GD] = D01 * A_xy + D22 * A_yx
            # KK[:, 1:KK.shape[1]:GD, 0:KK.shape[2]:GD] = D22 * A_yx + D01 * A_xy
        
        return KK
    
    @assemblymethod('fast_3d')
    def fast_assembly(self, space: _TS) -> TensorLike:
        index = self.index
        # coef = self.coef
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)

        if not isinstance(mesh, SimplexMesh):
            raise RuntimeError("The mesh should be an instance of SimplexMesh.")
        
        GD = mesh.geo_dimension()
        cm = mesh.entity_measure('cell', index=index)
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()

        # (NQ, LDOF, BC)
        gphi_lambda = scalar_space.grad_basis(bcs, index=index, variable='u')
        # (LDOF, LDOF, BC, BC)
        M = bm.einsum('q, qik, qjl->ijkl', ws, gphi_lambda, gphi_lambda)

        # (NC, LDOF, GD)
        glambda_x = mesh.grad_lambda()
        # (NC, LDOF, LDOF)
        A_xx = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 0], glambda_x[..., 0], cm)
        A_yy = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 1], glambda_x[..., 1], cm)
        A_zz = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 2], glambda_x[..., 2], cm)
        A_xy = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 0], glambda_x[..., 1], cm)
        A_xz = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 0], glambda_x[..., 2], cm)
        A_yx = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 1], glambda_x[..., 0], cm)
        A_yz = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 1], glambda_x[..., 2], cm)
        A_zx = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 2], glambda_x[..., 0], cm)
        A_zy = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 2], glambda_x[..., 1], cm)


        NC = mesh.number_of_cells()
        ldof = scalar_space.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64)

        D = self.material.elastic_matrix()
        if D.shape[1] != 1:
            raise ValueError("fast_assembly currently only supports elastic matrices "
                            "with shape (NC, 1, 6, 6) or (1, 1, 6, 6).")
        
        D00 = D[..., 0, 0, None]
        D01 = D[..., 0, 1, None]
        D55 = D[..., 5, 5, None]

        if space.dof_priority:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), D00 * A_xx + D55 * A_yy + D55 * A_zz)
            KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(ldof, 2 * ldof)), D00 * A_yy + D55 * A_xx + D55 * A_zz)
            KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(2 * ldof, None)), D00 * A_zz + D55 * A_xx + D55 * A_yy)
            # KK[:, :ldof, :ldof] = D00 * A_xx + D55 * A_yy + D55 * A_zz
            # KK[:, ldof:2*ldof, ldof:2*ldof] = D00 * A_yy + D55 * A_xx + D55 * A_zz
            # KK[:, 2*ldof:, 2*ldof:] = D00 * A_zz + D55 * A_xx + D55 * A_yy

            # Fill the off-diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, 2 * ldof)), D01 * A_xy + D55 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(2 * ldof, None)), D01 * A_xz + D55 * A_zx)
            KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(0, ldof)), D01 * A_yx + D55 * A_xy)
            KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(2 * ldof, None)), D01 * A_yz + D55 * A_zy)
            KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(0, ldof)), D01 * A_zx + D55 * A_xz)
            KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(ldof, 2 * ldof)), D01 * A_zy + D55 * A_yz)
            # KK[:, :ldof, ldof:2*ldof] = D01 * A_xy + D55 * A_yx
            # KK[:, :ldof, 2*ldof:] = D01 * A_xz + D55 * A_zx
            # KK[:, ldof:2*ldof, :ldof] = D01 * A_yx + D55 * A_xy
            # KK[:, ldof:2*ldof, 2*ldof:] = D01 * A_yz + D55 * A_zy
            # KK[:, 2*ldof:, :ldof] = D01 * A_zx + D55 * A_xz
            # KK[:, 2*ldof:, ldof:2*ldof] = D01 * A_zy + D55 * A_yz

        else:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), (2 * D55 + D01) * A_xx + D55 * (A_yy + A_zz))
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), (2 * D55 + D01) * A_yy + D55 * (A_xx + A_zz))
            KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(2, KK.shape[2], GD)), (2 * D55 + D01) * A_zz + D55 * (A_xx + A_yy))
            # KK[:, 0:KK.shape[1]:GD, 0:KK.shape[2]:GD] = (2 * D55 + D01) * A_xx + D55 * (A_yy + A_zz)
            # KK[:, 1:KK.shape[1]:GD, 1:KK.shape[2]:GD] = (2 * D55 + D01) * A_yy + D55 * (A_xx + A_zz)
            # KK[:, 2:KK.shape[1]:GD, 2:KK.shape[2]:GD] = (2 * D55 + D01) * A_zz + D55 * (A_xx + A_yy)

            # Fill the off-diagonal
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), D01 * A_xy + D55 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(2, KK.shape[2], GD)), D01 * A_xz + D55 * A_zx)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), D01 * A_yx + D55 * A_xy)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(2, KK.shape[2], GD)), D01 * A_yz + D55 * A_zy)
            KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(0, KK.shape[2], GD)), D01 * A_zx + D55 * A_xz)
            KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(1, KK.shape[2], GD)), D01 * A_zy + D55 * A_yz)
            # KK[:, 0:KK.shape[1]:GD, 1:KK.shape[2]:GD] = D01 * A_xy + D55 * A_yx
            # KK[:, 0:KK.shape[1]:GD, 2:KK.shape[2]:GD] = D01 * A_xz + D55 * A_zx
            # KK[:, 1:KK.shape[1]:GD, 0:KK.shape[2]:GD] = D01 * A_yx + D55 * A_xy
            # KK[:, 1:KK.shape[1]:GD, 2:KK.shape[2]:GD] = D01 * A_yz + D55 * A_zy
            # KK[:, 2:KK.shape[1]:GD, 0:KK.shape[2]:GD] = D01 * A_zx + D55 * A_xz
            # KK[:, 2:KK.shape[1]:GD, 1:KK.shape[2]:GD] = D01 * A_zy + D55 * A_yz

        return KK
