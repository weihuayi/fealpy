from typing import Optional, Literal
from ..mesh.mesh_base import SimplexMesh

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike

from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from ..decorator.variantmethod import variantmethod
from .integrator import LinearInt, OpInt, CellInt, enable_cache


class ScalarDiffusionIntegrator(LinearInt, OpInt, CellInt):
    r"""The diffusion integrator for function spaces based on homogeneous meshes."""
    def __init__(self, coef: Optional[CoefLike] = None, q: Optional[int] = None, *,
                 region: Optional[TensorLike] = None,
                 batched: bool = False,
                 method: Literal['fast', 'nonlinear', 'isopara', None] = None) -> None:
        super().__init__()
        self.coef = coef
        self.q = q
        self.set_region(region)
        self.batched = batched
        self.assembly.set(method)

    @enable_cache
    def to_global_dof(self, space: _FS, /, indices=None) -> TensorLike:
        return space.cell_to_dof(index=self.entity_selection(indices))

    @enable_cache
    def fetch_qf(self, space: _FS):
        mesh = space.mesh
        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        return bcs, ws

    @enable_cache
    def fetch_measure(self, space: _FS, /, indices=None):
        mesh = space.mesh
        return mesh.entity_measure('cell', index=self.entity_selection(indices))

    @enable_cache
    def fetch_gphix(self, space: _FS, /, indices=None):
        bcs = self.fetch_qf(space)[0]
        return space.grad_basis(bcs, index=self.entity_selection(indices), variable='x')

    @enable_cache
    def fetch_gphiu(self, space: _FS, /, indices=None):
        bcs = self.fetch_qf(space)[0]
        return space.grad_basis(bcs, index=self.entity_selection(indices), variable='u')

    @variantmethod
    def assembly(self, space: _FS, /, indices=None) -> TensorLike:
        coef = self.coef
        mesh = space.mesh
        bcs, ws = self.fetch_qf(space)
        cm = self.fetch_measure(space, indices)
        index = self.entity_selection(indices)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        gphi = self.fetch_gphix(space, indices)
        return bilinear_integral(gphi, gphi, ws, cm, coef, batched=self.batched)

    @assembly.register('fast')
    def assembly(self, space: _FS, /, indices=None) -> TensorLike:
        """
        限制：常系数、单纯形网格
        TODO: 加入 assert
        """
        mesh = space.mesh
        if isinstance(mesh, SimplexMesh):

            _, ws = self.fetch_qf(space)
            gphi = self.fetch_gphiu(space, indices)
            M = bm.einsum('q, qik, qjl -> ijkl', ws, gphi, gphi)
            cm = self.fetch_measure(space, indices)
            glambda = mesh.grad_lambda(index=self.entity_selection(indices))
            result = bm.einsum('ijkl, ckm, clm, c -> cij', M, glambda, glambda, cm)
        else:
            coef = self.coef
            mesh = space.mesh    
            index = self.entity_selection(indices)
            cm = self.fetch_measure(space, indices)
            bcs,ws = self.fetch_qf(space)
            coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
            
            gphiu = self.fetch_gphiu(space, indices)  
            M = bm.einsum('qim, qjn, q -> qijmn', gphiu, gphiu, ws)
            J = mesh.jacobi_matrix(bcs, index)
            G = mesh.first_fundamental_form(J)
            G = bm.linalg.inv(G)
            JG = bm.einsum("cqkm, cqmn -> cqkn", J, G) 
            result = bm.einsum('cqkn, qijmn, cqkm, c -> cij', JG, M, JG, cm) # (NC, NQ, ldof, GD)
        return result

    @assembly.register('nonlinear')
    def assembly(self, space: _FS, /, indices=None) -> TensorLike:
        uh = self.uh
        coef = self.coef
        mesh = space.mesh
        bcs, ws = self.fetch_qf(space)
        cm = self.fetch_measure(space, indices)
        gphi = self.fetch_gphix(space, indices)
        val_F = bm.squeeze(-uh.grad_value(bcs))   #(C, Q, dof_numel)
        index = self.entity_selection(indices)
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        coef_F = get_semilinear_coef(val_F, coef)
        return bilinear_integral(gphi, gphi, ws, cm, coef, batched=self.batched),\
               linear_integral(gphi, ws, cm, coef_F, batched=self.batched)

    @assembly.register('isopara')
    def assembly(self, space: _FS, /, indices=None) -> TensorLike:
        """
        曲面等参有限元积分子组装
        """
        index = self.entity_selection(indices)
        mesh = getattr(space, 'mesh', None)
        coef = self.coef
        rm = mesh.reference_cell_measure()
        cm = mesh.entity_measure('cell', index=index)

        q = space.p+3 if self.q is None else self.q
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = mesh.jacobi_matrix(bcs, index=index)
        G = mesh.first_fundamental_form(J) 
        d = bm.sqrt(bm.linalg.det(G))
        coef = process_coef_func(coef, bcs=bcs, mesh=mesh, etype='cell', index=index)
        gphi = space.grad_basis(bcs, index=index, variable='x')
        A = bm.einsum('q, cqim, cqjm, cq -> cij', ws*rm, gphi, gphi, d) * coef
        return A
