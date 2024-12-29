from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S

from ..mesh import HomogeneousMesh, SimplexMesh, TensorMesh
from ..functionspace.space import FunctionSpace as _FS
from ..functionspace.tensor_space import TensorFunctionSpace as _TS
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod
)
from fealpy.fem.utils import SymbolicIntegration

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
    def fetch_assembly(self, space: _FS):
        index = self.index
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
    
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
    
        cm = mesh.entity_measure('cell', index=index)
        q = scalar_space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = scalar_space.grad_basis(bcs, index=index, variable='x')

        if isinstance(mesh, TensorMesh):
            J = mesh.jacobi_matrix(bcs)
            detJ = bm.linalg.det(J)
        else:
            J = None
            detJ = None
            
        return cm, bcs, ws, gphi, detJ

    @enable_cache
    def fetch_voigt_assembly(self, space: _FS):
        index = self.index
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
    
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
    
        cm = mesh.entity_measure('cell', index=index)
        q = scalar_space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = scalar_space.grad_basis(bcs, index=index, variable='x')

        if isinstance(mesh, TensorMesh):
            J = mesh.jacobi_matrix(bcs)
            detJ = bm.linalg.det(J)
        else:
            J = None
            detJ = None

        D = self.material.elastic_matrix(bcs)
        B = self.material.strain_matrix(dof_priority=space.dof_priority, 
                                        gphi=gphi)
            
        return cm, ws, detJ, D, B
    
    @enable_cache
    def fetch_voigt_assembly_uniform(self, space: _FS):
        index = self.index
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
    
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
    
        cm = mesh.entity_measure('cell', index=index)
        q = scalar_space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = scalar_space.grad_basis(bcs, index=index, variable='x')

        D = self.material.elastic_matrix(bcs)
        B = self.material.strain_matrix(dof_priority=space.dof_priority, 
                                        gphi=gphi)
            
        return cm, ws, D, B

    @enable_cache
    def fetch_fast_assembly(self, space: _FS):
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
        gphi_lambda = space.grad_basis(bcs, index=index, variable='u')    # (NQ, LDOF, BC)

        if isinstance(mesh, SimplexMesh):
            glambda_x = mesh.grad_lambda()       # (NC, LDOF, GD)
            J = None
            G = None
            JG = None
        else:
            glambda_x = None
            J = mesh.jacobi_matrix(bcs)                   # (NC, NQ, GD, GD)
            G = mesh.first_fundamental_form(J)            # (NC, NQ, GD, GD)
            G = bm.linalg.inv(G)                          # (NC, NQ, GD, GD)
            JG = bm.einsum('cqkm, cqmn -> cqkn', J, G)    # (NC, NQ, GD, GD)

        return cm, ws, gphi_lambda, glambda_x, JG
    
    @enable_cache
    def fetch_fast_assembly_uniform(self, space: _FS):
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
        gphi_lambda = space.grad_basis(bcs, index=index, variable='u')    # (NQ, LDOF, BC)

        J = mesh.jacobi_matrix(bcs)[:, 0, ...]        # (NC, GD, GD)
        G = mesh.first_fundamental_form(J)            # (NC, GD, GD)
        G = bm.linalg.inv(G)                          # (NC, GD, GD)
        JG = bm.einsum('ckm, cmn -> ckn', J, G)       # (NC, GD, GD)

        M = bm.einsum('qim, qjn, q -> ijmn', gphi_lambda, gphi_lambda, ws)  # (LDOF, LDOF, BC, BC)

        return cm, JG, M
    
    @enable_cache
    def fetch_symbolic_assembly(self, space: _TS) -> TensorLike:
        index = self.index
        mesh = getattr(space, 'mesh', None)
    
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
    
        cm = mesh.entity_measure('cell', index=index)
        glambda_x = mesh.grad_lambda()  # (NC, LDOF, GD)
        
        symbolic_int = SymbolicIntegration(space)
        M = bm.tensor(symbolic_int.gphi_gphi_matrix()) # (LDOF1, LDOF1, GD+1, GD+1)

        return cm, mesh, glambda_x, bm.asarray(M, dtype=bm.float64)
    
    @enable_cache
    def fetch_c3d8_bbar_assembly(self, space: _FS):
        index = self.index
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
    
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
    
        cm = mesh.entity_measure('cell', index=index)
        q = scalar_space.p+1 if self.q is None else self.q
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = scalar_space.grad_basis(bcs, index=index, variable='x')

        J = mesh.jacobi_matrix(bcs)
        detJ = bm.linalg.det(J)

        D = self.material.elastic_matrix(bcs)
        B = self.material.strain_matrix(dof_priority=space.dof_priority, 
                                        gphi=gphi, 
                                        correction='BBar', 
                                        cm=cm, ws=ws, detJ=detJ)
            
        return ws, detJ, D, B

    def assembly(self, space: _TS) -> TensorLike:
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
        cm, bcs, ws, gphi, detJ = self.fetch_assembly(space)

        if isinstance(mesh, TensorMesh):
            J = mesh.jacobi_matrix(bcs)
            detJ = bm.linalg.det(J)
            A_xx = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 0], gphi[..., 0], detJ)
            A_yy = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 1], gphi[..., 1], detJ)
            A_xy = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 0], gphi[..., 1], detJ)
            A_yx = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 1], gphi[..., 0], detJ)
        else:
            A_xx = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 0], gphi[..., 0], cm)
            A_yy = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 1], gphi[..., 1], cm)
            A_xy = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 0], gphi[..., 1], cm)
            A_yx = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 1], gphi[..., 1], cm)

        GD = mesh.geo_dimension()
        if GD == 3:
            if isinstance(mesh, TensorMesh):
                A_xz = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 0], gphi[..., 2], detJ)
                A_zx = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 2], gphi[..., 0], detJ)
                A_yz = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 1], gphi[..., 2], detJ)
                A_zy = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 2], gphi[..., 1], detJ)
                A_zz = bm.einsum('q, cqi, cqj, cq -> cij', ws, gphi[..., 2], gphi[..., 2], detJ)
            else:
                A_xz = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 0], gphi[..., 2], cm)
                A_zx = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 2], gphi[..., 0], cm)
                A_yz = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 1], gphi[..., 2], cm)
                A_zy = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 2], gphi[..., 1], cm)
                A_zz = bm.einsum('q, cqi, cqj, c -> cij', ws, gphi[..., 2], gphi[..., 2], cm)

        D = self.material.elastic_matrix(bcs)
        if D.shape[1] != 1:
            raise ValueError("assembly currently only supports elastic matrices "
                            f"with shape (NC, 1, {2*GD}, {2*GD}) or (1, 1, {2*GD}, {2*GD}).")

        NC = mesh.number_of_cells()
        ldof = scalar_space.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64)

        if GD == 2:
            D00 = D[..., 0, 0, None]  # 2D: E/(1-ν²) 或 2μ+λ
            D01 = D[..., 0, 1, None]  # 2D: νE/(1-ν²) 或 λ
            D22 = D[..., 2, 2, None]  # 2D: E/2(1+ν) 或 μ

            if space.dof_priority:
                # 填充对角块
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), 
                            D00 * A_xx + D22 * A_yy)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(ldof, None)), 
                            D00 * A_yy + D22 * A_xx)
                
                # 填充非对角块
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, None)), 
                            D01 * A_xy + D22 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(ldof, None), slice(0, ldof)), 
                            D01 * A_yx + D22 * A_xy)
            else:
                # 填充对角块
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                            D00 * A_xx + D22 * A_yy)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                            D00 * A_yy + D22 * A_xx)
                
                # 填充非对角块
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                            D01 * A_xy + D22 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                            D01 * A_yx + D22 * A_xy)

        else: 
            D00 = D[..., 0, 0, None]  # 2μ + λ
            D01 = D[..., 0, 1, None]  # λ
            D55 = D[..., 5, 5, None]  # μ

            if space.dof_priority:
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), 
                            D00 * A_xx + D55 * (A_yy + A_zz))
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(ldof, 2 * ldof)), 
                            D00 * A_yy + D55 * (A_xx + A_zz))
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(2 * ldof, None)), 
                            D00 * A_zz + D55 * (A_xx + A_yy))

                # 填充非对角块
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, 2 * ldof)), 
                            D01 * A_xy + D55 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(2 * ldof, None)), 
                            D01 * A_xz + D55 * A_zx)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(0, ldof)), 
                            D01 * A_yx + D55 * A_xy)
                KK = bm.set_at(KK, (slice(None), slice(ldof, 2 * ldof), slice(2 * ldof, None)), 
                            D01 * A_yz + D55 * A_zy)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(0, ldof)), 
                            D01 * A_zx + D55 * A_xz)
                KK = bm.set_at(KK, (slice(None), slice(2 * ldof, None), slice(ldof, 2 * ldof)), 
                            D01 * A_zy + D55 * A_yz)
            else:
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                            (2 * D55 + D01) * A_xx + D55 * (A_yy + A_zz))
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                            (2 * D55 + D01) * A_yy + D55 * (A_xx + A_zz))
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(2, KK.shape[2], GD)), 
                            (2 * D55 + D01) * A_zz + D55 * (A_xx + A_yy))

                # 填充非对角块
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                            D01 * A_xy + D55 * A_yx)
                KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(2, KK.shape[2], GD)), 
                            D01 * A_xz + D55 * A_zx)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                            D01 * A_yx + D55 * A_xy)
                KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(2, KK.shape[2], GD)), 
                            D01 * A_yz + D55 * A_zy)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                            D01 * A_zx + D55 * A_xz)
                KK = bm.set_at(KK, (slice(None), slice(2, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                            D01 * A_zy + D55 * A_yz)

        return KK 
    
    @assemblymethod('voigt')
    def voigt_assembly(self, space: _TS) -> TensorLike:
        mesh = getattr(space, 'mesh', None)
        cm, ws, detJ, D, B = self.fetch_voigt_assembly(space)
        
        if isinstance(mesh, TensorMesh):
            KK = bm.einsum('q, cq, cqki, cqkl, cqlj -> cij',
                            ws, detJ, B, D, B)
        else:
            KK = bm.einsum('q, c, cqki, cqkl, cqlj, cq -> cij',
                            ws, cm, B, D, B)
            
        return KK
    
    @assemblymethod('voigt_uniform')
    def voigt_assembly_uniform(self, space: _TS) -> TensorLike:
        cm, ws, D, B = self.fetch_voigt_assembly_uniform(space)
        
        KK = bm.einsum('q, c, cqki, cqkl, cqlj -> cij', ws, cm, B, D, B)
            
        return KK
    
    @assemblymethod('fast_strain')
    def fast_assembly_strain(self, space: _TS) -> TensorLike:
        scalar_space = space.scalar_space
        cm, ws, mesh, gphi_lambda, glambda_x = self.fetch_fast_assembly(scalar_space)

        if not isinstance(mesh, SimplexMesh):
            raise RuntimeError("The mesh should be an instance of SimplexMesh.")

        GD = mesh.geo_dimension()

        # (LDOF, LDOF, BC, BC)
        M = bm.einsum('q, qik, qjl -> ijkl', ws, gphi_lambda, gphi_lambda)

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
        D00 = D[..., 0, 0, None]  # 2*\mu + \lambda
        D01 = D[..., 0, 1, None]  # \lambda
        D22 = D[..., 2, 2, None]  # \mu
        
        if space.dof_priority:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof, 1), slice(0, ldof, 1)), D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1], 1), slice(ldof, KK.shape[1], 1)), D00 * A_yy + D22 * A_xx)

            # Fill the off-diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof, 1), slice(ldof, KK.shape[1], 1)), D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1], 1), slice(0, ldof, 1)), D01 * A_yx + D22 * A_xy)
        else:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), D00 * A_yy + D22 * A_xx)

            # Fill the off-diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), D01 * A_yx + D22 * A_xy)

        
        return KK
    
    @assemblymethod('fast_stress')
    def fast_assembly_stress(self, space: _TS) -> TensorLike:
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
        cm, ws, gphi_lambda, glambda_x, JG = self.fetch_fast_assembly(scalar_space)
        
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        ldof = scalar_space.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64)

        D = self.material.elastic_matrix()
        if D.shape[1] != 1:
            raise ValueError("fast_assembly_stress currently only supports elastic matrices "
                            "with shape (NC, 1, 3, 3) or (1, 1, 3, 3).")
        D00 = D[..., 0, 0, None]  # E / (1-\nu^2) * 1
        D01 = D[..., 0, 1, None]  # E / (1-\nu^2) * \nu
        D22 = D[..., 2, 2, None]  # E / (1-\nu^2) * (1-nu)/2

        if isinstance(mesh, SimplexMesh):
            M = bm.einsum('q, qik, qjl -> ijkl', ws, gphi_lambda, gphi_lambda)  # (LDOF, LDOF, BC, BC)
            A_xx = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 0], glambda_x[..., 0], cm)
            A_yy = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 1], glambda_x[..., 1], cm)
            A_xy = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 0], glambda_x[..., 1], cm)
            A_yx = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 1], glambda_x[..., 0], cm)
        else:
            M = bm.einsum('qim, qjn, q -> ijmnq', gphi_lambda, gphi_lambda, ws)  # (LDOF, LDOF, GD, GD, NQ)
            A_xx = bm.einsum('ijmnq, cqm, cqn, c -> cij', M, JG[...,0,:], JG[...,0,:], cm) # (NC, LDOF, LDOF)
            A_yy = bm.einsum('ijmnq, cqm, cqn, c -> cij', M, JG[...,1,:], JG[...,1,:], cm) # (NC, LDOF, LDOF)
            A_xy = bm.einsum('ijmnq, cqm, cqn, c -> cij', M, JG[...,0,:], JG[...,1,:], cm) # (NC, LDOF, LDOF)
            A_yx = bm.einsum('ijmnq, cqm, cqn, c -> cij', M, JG[...,1,:], JG[...,0,:], cm) # (NC, LDOF, LDOF)
            # A = bm.einsum('ijmnq, cqam, cqbn, c -> cijab', M, JG, JG, cm)    # (NC, ldof, ldof, GD, GD)
            # A_xx = A[..., 0, 0]  # (NC, LDOF, LDOF)
            # A_yy = A[..., 1, 1]  # (NC, LDOF, LDOF)
            # A_xy = A[..., 0, 1]  # (NC, LDOF, LDOF)
            # A_yx = A[..., 1, 0]  # (NC, LDOF, LDOF)
        
        # 填充刚度矩阵
        if space.dof_priority:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), 
                        D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(ldof, KK.shape[1])), 
                        D00 * A_yy + D22 * A_xx)

            # Fill the off-diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, KK.shape[1])), 
                        D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(0, ldof)), 
                        D01 * A_yx + D22 * A_xy)
        else:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                        D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                        D00 * A_yy + D22 * A_xx)

            # Fill the off-diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                        D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                        D01 * A_yx + D22 * A_xy)

        return KK
    
    @assemblymethod('fast_stress_uniform')
    def fast_assembly_stress_uniform(self, space: _TS) -> TensorLike:
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
        cm, JG, M = self.fetch_fast_assembly_uniform(scalar_space)
        
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        ldof = scalar_space.number_of_local_dofs()
        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64)

        D = self.material.elastic_matrix()
        if D.shape[1] != 1:
            raise ValueError("fast_assembly_stress currently only supports elastic matrices "
                            "with shape (NC, 1, 3, 3) or (1, 1, 3, 3).")
        D00 = D[..., 0, 0]  # E / (1-\nu^2) * 1
        D01 = D[..., 0, 1]  # E / (1-\nu^2) * \nu
        D22 = D[..., 2, 2]  # E / (1-\nu^2) * (1-nu)/2

        # A = bm.einsum('ijmn, cam, cbn, c -> cijab', M, JG, JG, cm)  # (NC, LDOF, LDOF, GD, GD)
        # A_xx = A[..., 0, 0]
        # A_yy = A[..., 1, 1]
        # A_xy = A[..., 0, 1]
        # A_yx = A[..., 1, 0]
        A_xx = bm.einsum('ijmn, cm, cn, c -> cij', M, JG[..., 0], JG[..., 0], cm)  # (NC, LDOF, LDOF)
        A_yy = bm.einsum('ijmn, cm, cn, c -> cij', M, JG[..., 1], JG[..., 1], cm)  # (NC, LDOF, LDOF)
        A_xy = bm.einsum('ijmn, cm, cn, c -> cij', M, JG[..., 0], JG[..., 1], cm)  # (NC, LDOF, LDOF)
        A_yx = bm.einsum('ijmn, cm, cn, c -> cij', M, JG[..., 1], JG[..., 0], cm)  # (NC, LDOF, LDOF)
        
        # 填充刚度矩阵
        if space.dof_priority:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), 
                        D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(ldof, KK.shape[1])), 
                        D00 * A_yy + D22 * A_xx)

            # Fill the off-diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, KK.shape[1])), 
                        D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(0, ldof)), 
                        D01 * A_yx + D22 * A_xy)
        else:
            # Fill the diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                        D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                        D00 * A_yy + D22 * A_xx)

            # Fill the off-diagonal part
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                        D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                        D01 * A_yx + D22 * A_xy)

        return KK
    
    @assemblymethod('symbolic_stress')
    def symbolic_assembly_stress(self, space: _TS) -> TensorLike:
        scalar_space = space.scalar_space
        cm, mesh, glambda_x, M = self.fetch_symbolic_assembly(scalar_space)

        if not isinstance(mesh, SimplexMesh):
            raise RuntimeError("The mesh should be an instance of SimplexMesh.")
        
        GD = mesh.geo_dimension()
        NC = mesh.number_of_cells()
        ldof = scalar_space.number_of_local_dofs()

        # 计算各方向的矩阵
        A_xx = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 0], glambda_x[..., 0], cm)
        A_yy = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 1], glambda_x[..., 1], cm)
        A_xy = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 0], glambda_x[..., 1], cm)
        A_yx = bm.einsum('ijkl, ck, cl, c -> cij', M, glambda_x[..., 1], glambda_x[..., 0], cm)

        KK = bm.zeros((NC, GD * ldof, GD * ldof), dtype=bm.float64)

        # 获取材料矩阵
        D = self.material.elastic_matrix()
        if D.shape[1] != 1:
            raise ValueError("symbolic_assembly currently only supports elastic matrices "
                            "with shape (NC, 1, 3, 3) or (1, 1, 3, 3).")
        
        D00 = D[..., 0, 0, None]  # 2μ + λ
        D01 = D[..., 0, 1, None]  # λ
        D22 = D[..., 2, 2, None]  # μ

        if space.dof_priority:
            # 填充对角块
            KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(0, ldof)), 
                        D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(ldof, KK.shape[1])), 
                        D00 * A_yy + D22 * A_xx)
            
            # 填充非对角块
            KK = bm.set_at(KK, (slice(None), slice(0, ldof), slice(ldof, KK.shape[1])), 
                        D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(ldof, KK.shape[1]), slice(0, ldof)), 
                        D01 * A_yx + D22 * A_xy)
        else:
            # 类似的填充方式，但使用不同的索引方式
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                        D00 * A_xx + D22 * A_yy)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                        D00 * A_yy + D22 * A_xx)
            
            KK = bm.set_at(KK, (slice(None), slice(0, KK.shape[1], GD), slice(1, KK.shape[2], GD)), 
                        D01 * A_xy + D22 * A_yx)
            KK = bm.set_at(KK, (slice(None), slice(1, KK.shape[1], GD), slice(0, KK.shape[2], GD)), 
                        D01 * A_yx + D22 * A_xy)

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
    
    @assemblymethod('C3D8_BBar')
    def c3d8_bbar_assembly(self, space: _TS) -> TensorLike:
        ws, detJ, D, B  = self.fetch_c3d8_bbar_assembly(space)
        
        KK = bm.einsum('q, cq, cqki, cqkl, cqlj -> cij',
                        ws, detJ, B, D, B)

        return KK
