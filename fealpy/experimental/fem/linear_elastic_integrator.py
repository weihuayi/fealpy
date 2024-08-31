from typing import Optional

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, MaterialLike

from .utils import shear_strain, normal_strain
from ..utils import process_coef_func, is_scalar, is_tensor
from ..functionspace.utils import flatten_indices

from ..mesh import HomogeneousMesh, SimplexMesh
from ..functionspace.space import FunctionSpace as _FS
from .integrator import (
    LinearInt, OpInt, CellInt,
    enable_cache,
    assemblymethod,
    CoefLike
)

from ..material.elastic_material import LinearElasticMaterial


class LinearElasticIntegrator(LinearInt, OpInt, CellInt):
    """
    The linear elastic integrator for function spaces based on homogeneous meshes.
    """
    def __init__(self, 
                 material: MaterialLike,
                 elastic_type: Optional[str]=None,
                 coef: Optional[CoefLike]=None, q: Optional[int]=None, *,
                 index: Index=_S,
                 method: Optional[str]=None) -> None:
        method = 'assembly' if (method is None) else method
        super().__init__(method=method)

        self.material = material  
        self.elastic_type = elastic_type
        self.coef = coef
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
    
    def elastic_matrix(self, space: _FS):
        elastic_type = self.elastic_type
        scalar_space = space.scalar_space
        _, _, gphi, _, _, _ = self.fetch(scalar_space)
        _, GD = gphi.shape[-2:]

        if GD == 2:
            if elastic_type == 'stress':
                D = self.material.plane_stress_elastic_matrix()
            elif elastic_type == 'strain':
                D = self.material.plane_strain_elastic_matrix()
            else:
                raise ValueError("Unknown elastic type for 2D. Use 'stress' or 'strain'.")
        elif GD == 3:
            if elastic_type is None:
                D = self.material.elastic_matrix()
            else:
                raise ValueError("For 3D elastic matrix, 'elastic_type' should not be specified.")
        else:
            raise ValueError("Invalid GD dimension.")
        
        return D
    
    def strain_matrix(self, space: _FS) -> TensorLike:
        '''
        GD = 2: (NC, NQ, 3, tldof)
        GD = 3: (NC, NQ, 6, tldof)
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
        
        D = self.elastic_matrix(space)
        B = self.strain_matrix(space)
        if coef is None:
            KK = bm.einsum('q, c, cqki, kl, cqlj -> cij', ws, cm, B, D, B)
        elif is_scalar(coef):
            KK = coef * bm.einsum('q, c, cqki, kl, cqlj -> cij', ws, cm, B, D, B)
        elif is_tensor(coef):
            if coef.ndim == 1:
                KK = bm.einsum('q, c, cqki, kl, cqlj, c -> cij', ws, cm, B, D, B, coef)
            elif coef.ndim == 2:
                KK = bm.einsum('q, c, cqki, kl, cqlj, cq -> cij', ws, cm, B, D, B, coef)
            elif coef.ndim == 3:
                pass
            else:
                raise ValueError("Invalid coef.")
        
        return KK

    @assemblymethod('fast_strain')
    def fast_assembly_strain(self, space: _FS) -> TensorLike:
        index = self.index
        coef = self.coef
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

        mu, lam = self.material.calculate_shear_modulus(), self.material.calculate_lame_lambda()
        
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
            KK[:] = KK
        elif is_scalar(coef):
            KK[:] = KK * coef
        elif is_tensor(coef):
            if coef.ndim == 1:
                KK[:] = bm.einsum('cij, c -> cij', KK, coef)
            elif coef.ndim == 2:
                raise ValueError("Invalid coef shape: \
                        Fast assembly expects coef to be of shape (NC, NQ).")
            elif coef.ndim == 3:
                pass
            else:
                raise ValueError("Invalid coef.")
        
        return KK

    @assemblymethod('fast_stress')
    def fast_assembly_stress(self, space: _FS) -> TensorLike:
        index = self.index
        coef = self.coef
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

        E, nu = self.material.get_property('elastic_modulus'), self.material.get_property('poisson_ratio')
        # Fill the diagonal part
        KK[:, :ldof, :ldof] = A_xx + (1 - nu) / 2 * A_yy
        KK[:, ldof:, ldof:] = A_yy + (1 - nu) / 2 * A_xx

        # Fill the off-diagonal part
        KK[:, :ldof, ldof:] = nu * A_xy + (1 - nu) / 2 * A_yx
        KK[:, ldof:, :ldof] = (1 - nu) / 2 * A_yx + nu * A_xy

        KK *= E / (1 - nu**2)

        if coef is None:
            KK[:] = KK
        elif is_scalar(coef):
            KK[:] = KK * coef
        elif is_tensor(coef):
            if coef.ndim == 1:
                KK[:] = bm.einsum('cij, c -> cij', KK, coef)
            elif coef.ndim == 2:
                raise ValueError("Invalid coef shape: \
                        Fast assembly expects coef to be of shape (NC, NQ).")
            elif coef.ndim == 3:
                pass
            else:
                raise ValueError("Invalid coef.")
        
        return KK
    
    @assemblymethod('fast_3d')
    def fast_assembly(self, space: _FS) -> TensorLike:
        index = self.index
        coef = self.coef
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

        mu, lam = self.material.calculate_shear_modulus(), self.material.calculate_lame_lambda()
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
            KK[:] = KK
        elif is_scalar(coef):
            KK[:] = KK * coef
        elif is_tensor(coef):
            if coef.ndim == 1:
                KK[:] = bm.einsum('cij, c -> cij', KK, coef)
            elif coef.ndim == 2:
                raise ValueError("Invalid coef shape: \
                        Fast assembly expects coef to be of shape (NC, NQ).")
            elif coef.ndim == 3:
                pass
            else:
                raise ValueError("Invalid coef.")
        
        return KK
