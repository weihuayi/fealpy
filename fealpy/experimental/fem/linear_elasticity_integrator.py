from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S

from .utils import shear_strain, normal_strain
from ..utils import process_coef_func, is_scalar, is_tensor
from ..functionspace.utils import flatten_indices


from ..mesh import HomogeneousMesh, SimplexMesh, StructuredMesh
from ..functionspace.space import FunctionSpace as _FS
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod,
    CoefLike
)


class LinearElasticityIntegrator(LinearInt, OpInt, CellInt):
    """
    The linear elasticity integrator for function spaces based on homogeneous meshes.
    """
    def __init__(self, 
                 lam: Optional[float]=None, mu: Optional[float]=None,
                 E: Optional[float]=None, nu: Optional[float]=None, 
                 elasticity_type: Optional[str]=None,
                 coef: Optional[CoefLike]=None, q: int=5, *,
                 index: Index=_S,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)
        if lam is not None and mu is not None:
            self.E = mu * (3*lam + 2*mu) / (lam + mu)
            self.nu = lam / (2 * (lam + mu))
            self.lam = lam
            self.mu = mu
        elif E is not None and nu is not None:
            self.lam = nu * E / ((1 + nu) * (1 - 2 * nu))
            self.mu = E / (2 * (1 + nu))
            self.E = E
            self.nu = nu
        else:
            raise ValueError("Either (lam, mu) or (e, nu) should be provided.")
        self.elasticity_type = elasticity_type
        self.coef = coef
        self.q = q
        self.index = index

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)
    
        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The LinearElasticityIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")
    
        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.quadrature_formula(q)
        bcs, ws = qf.get_quadrature_points_and_weights()
        gphi = space.grad_basis(bcs, index=index, variable='x')
        return bcs, ws, gphi, cm, index, q
    
    def elasticity_matrix(self, space: _FS):
        elasticity_type = self.elasticity_type
        scalar_space = space.scalar_space
        _, _, gphi, _, _, _ = self.fetch(scalar_space)
        _, GD = gphi.shape[-2:]

        if GD == 2:
            if elasticity_type == 'stress':
                E, nu = self.E, self.nu
                D = E / (1 - nu**2) * \
                    bm.tensor([[1, nu, 0],
                                [nu, 1, 0],
                                [0, 0, (1 - nu) / 2]], dtype=bm.float64)
            elif elasticity_type == 'strain':
                mu, lam = self.mu, self.lam
                D = bm.tensor([[2 * mu + lam, lam, 0],
                                  [lam, 2 * mu + lam, 0],
                                  [0, 0, mu]], dtype=bm.float64)
            else:
                raise ValueError("Unknown type.")
        elif GD == 3:
            if elasticity_type is None:
                mu, lam = self.mu, self.lam
                D = bm.tensor([[2 * mu + lam, lam, lam, 0, 0, 0],
                                  [lam, 2 * mu + lam, lam, 0, 0, 0],
                                  [lam, lam, 2 * mu + lam, 0, 0, 0],
                                  [0, 0, 0, mu, 0, 0],
                                  [0, 0, 0, 0, mu, 0],
                                  [0, 0, 0, 0, 0, mu]], dtype=bm.float64)
            else:
                raise ValueError("Unnecessary Input.")
        else:
            raise ValueError("Invalid GD dimension.")
        
        return D
    
    def strain_matrix(self, space: _FS) -> TensorLike:
        '''
        GD = 2: (NC, NQ, 3, tldof)
        GD = 2: (NC, NQ, 6, tldof)
        '''
        scalar_space = space.scalar_space
        _, _, gphi, _, _, _ = self.fetch(scalar_space)
        ldof, GD = gphi.shape[-2:]
        if space.dof_priority:
            indices = flatten_indices((ldof, GD), (1, 0))
        else:
            indices = flatten_indices((ldof, GD), (0, 1))
        B = bm.concat([normal_strain(gphi, indices),
                       shear_strain(gphi, indices)], axis=-2)
        return B
    
    def assembly(self, space: _FS) -> TensorLike:
        coef = self.coef
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)
        bcs, ws, _, cm, index, _ = self.fetch(scalar_space)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        D = self.elasticity_matrix(space)
        B = self.strain_matrix(space)
        if coef is None:
            KK = bm.einsum('q, c, cqki, kl, cqlj -> cij', ws, cm, B, D, B)
        elif is_scalar(coef):
            KK = coef * bm.einsum('q, c, cqki, kl, cqlj -> cij', ws, cm, B, D, B)
        elif is_tensor(coef):
            # TODO coef 现在只能为一阶张量
            KK = bm.einsum('q, c, cqki, kl, cqlj, c -> cij', ws, cm, B, D, B, coef)
        
        return KK

    @assemblymethod('fast_strain')
    def fast_assembly_strain_constant(self, space: _FS) -> TensorLike:
        q = self.q
        index = self.index
        coef = self.coef
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)

        if not isinstance(mesh, SimplexMesh):
            raise RuntimeError("The mesh should be an instance of SimplexMesh.")

        GD = mesh.geo_dimension()
        cm = mesh.entity_measure('cell', index=index)
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

        mu, lam = self.mu, self.lam
        
        if space.dof_priority:
            # Fill the diagonal part
            KK[:, 0:ldof:1, 0:ldof:1] = (2 * mu + lam) * A_xx + mu * A_yy
            KK[:, ldof:KK.shape[1]:1, ldof:KK.shape[1]:1] = (2 * mu + lam) * A_yy + mu * A_xx

            # Fill the off-diagonal part
            KK[:, 0:ldof:1, ldof:KK.shape[1]:1] = lam * A_xy + mu * A_yx
            KK[:, ldof:KK.shape[1]:1, 0:ldof:1] = lam * A_yx + mu * A_xy
        else:
            # Fill the diagonal part
            KK[:, 0:KK.shape[1]:GD, 0:KK.shape[2]:GD] = (2 * mu + lam) * A_xx + mu * A_yy
            KK[:, GD-1:KK.shape[1]:GD, GD-1:KK.shape[2]:GD] = (2 * mu + lam) * A_yy + mu * A_xx

            # Fill the off-diagonal part
            KK[:, 0:KK.shape[1]:GD, GD-1:KK.shape[2]:GD] = lam * A_xy + mu * A_yx
            KK[:, GD-1:KK.shape[1]:GD, 0:KK.shape[2]:GD] = lam * A_yx + mu * A_xy

        if coef is None:
            return KK
        
        if is_scalar(coef):
            KK[:] = KK * coef
            return KK
        elif is_tensor(coef):
            KK[:] = bm.einsum('cij, c -> cij', KK, coef)
            return KK
        

    @assemblymethod('fast_stress')
    def fast_assembly_stress_constant(self, space: _FS) -> TensorLike:
        q = self.q
        index = self.index
        coef = self.coef
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)

        if not isinstance(mesh, SimplexMesh):
            raise RuntimeError("The mesh should be an instance of SimplexMesh.")
        
        GD = mesh.geo_dimension()
        cm = mesh.entity_measure('cell', index=index)
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

        E, nu = self.E, self.nu
        # Fill the diagonal part
        KK[:, :ldof, :ldof] = A_xx + (1 - nu) / 2 * A_yy
        KK[:, ldof:, ldof:] = A_yy + (1 - nu) / 2 * A_xx

        # Fill the off-diagonal part
        KK[:, :ldof, ldof:] = nu * A_xy + (1 - nu) / 2 * A_yx
        KK[:, ldof:, :ldof] = (1 - nu) / 2 * A_yx + nu * A_xy

        KK *= E / (1 - nu**2)

        if coef is None:
            return KK
        
        if is_scalar(coef):
            KK[:] = KK * coef
            return KK
        elif is_tensor(coef):
            KK[:] = bm.einsum('cij, c -> cij', KK, coef)
            return KK
    
    @assemblymethod('fast_3d')
    def fast_assembly_constant(self, space: _FS) -> TensorLike:
        q = self.q
        index = self.index
        coef = self.coef
        scalar_space = space.scalar_space
        mesh = getattr(scalar_space, 'mesh', None)

        if not isinstance(mesh, SimplexMesh):
            raise RuntimeError("The mesh should be an instance of SimplexMesh.")
        
        GD = mesh.geo_dimension()
        cm = mesh.entity_measure('cell', index=index)
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

        mu, lam = self.mu, self.lam
        # Fill the diagonal part
        KK[:, :ldof, :ldof] = (2 * mu + lam) * A_xx + mu * (A_yy + A_zz)
        KK[:, ldof:2*ldof, ldof:2*ldof] = (2 * mu + lam) * A_yy + mu * (A_xx + A_zz)
        KK[:, 2*ldof:, 2*ldof:] = (2 * mu + lam) * A_zz + mu * (A_xx + A_yy)

        # Fill the off-diagonal part
        KK[:, :ldof, ldof:2*ldof] = lam * A_xy + mu * A_yx
        KK[:, :ldof, 2*ldof:] = lam * A_xz + mu * A_zx
        KK[:, ldof:2*ldof, :ldof] = lam * A_yx + mu * A_xy
        KK[:, ldof:2*ldof, 2*ldof:] = lam * A_yz + mu * A_zy
        KK[:, 2*ldof:, :ldof] = lam * A_zx + mu * A_xz
        KK[:, 2*ldof:, ldof:2*ldof] = lam * A_zy + mu * A_yz

        if coef is None:
            return KK
        
        if is_scalar(coef):
            KK[:] = KK * coef
            return KK
        elif is_tensor(coef):
            KK[:] = bm.einsum('cij, c -> cij', KK, coef)
            return KK


class LinearElasticityCoefficient():
    def __init__(self, lam: Optional[float]=None, mu: Optional[float]=None,
                 E: Optional[float]=None, nu: Optional[float]=None, 
                 elasticity_type: Optional[str]=None) -> None:
        if lam is not None and mu is not None:
            self.E = mu * (3*lam + 2*mu) / (lam + mu)
            self.nu = lam / (2 * (lam + mu))
            self.lam = lam
            self.mu = mu
        elif E is not None and nu is not None:
            self.lam = nu * E / ((1 + nu) * (1 - 2 * nu))
            self.mu = E / (2 * (1 + nu))
            self.E = E
            self.nu = nu
        else:
            raise ValueError("Either (lam, mu) or (e, nu) should be provided.")
        self.elasticity_type = elasticity_type

    def elasticity_matrix(self, space: _FS):
        elasticity_type = self.elasticity_type
        scalar_space = space.scalar_space
        _, _, gphi, _, _, _ = self.fetch(scalar_space)
        _, GD = gphi.shape[-2:]

        if GD == 2:
            if elasticity_type == 'stress':
                E, nu = self.E, self.nu
                D = E / (1 - nu**2) *\
                    bm.tensor([[1, nu, 0],
                                  [nu, 1, 0],
                                  [0, 0, (1 - nu) / 2]], device=self.device, dtype=bm.float64)
            elif elasticity_type == 'strain':
                mu, lam = self.mu, self.lam
                D = bm.tensor([[2 * mu + lam, lam, 0],
                                  [lam, 2 * mu + lam, 0],
                                  [0, 0, mu]], device=self.device, dtype=bm.float64)
            else:
                raise ValueError("Unknown type.")
        elif GD == 3:
            if elasticity_type is None:
                D = bm.tensor([[2 * mu + lam, lam, lam, 0, 0, 0],
                                  [lam, 2 * mu + lam, lam, 0, 0, 0],
                                  [lam, lam, 2 * mu + lam, 0, 0, 0],
                                  [0, 0, 0, mu, 0, 0],
                                  [0, 0, 0, 0, mu, 0],
                                  [0, 0, 0, 0, 0, mu]], dtype=bm.float64)
            else:
                raise ValueError("Unnecessary Input.")
        else:
            raise ValueError("Invalid GD dimension.")
        
        return D
    
    def strain_matrix(self, space: _FS):
        scalar_space = space.scalar_space
        _, _, gphi, _, _, _ = self.fetch(scalar_space)
        ldof, GD = gphi.shape[-2:]
        if space.dof_priority:
            indices = flatten_indices((ldof, GD), (1, 0))
        else:
            indices = flatten_indices((ldof, GD), (0, 1))
        B = bm.cat([normal_strain(gphi, indices),
                       shear_strain(gphi, indices)], dim=-2)
        return B