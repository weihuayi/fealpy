from typing import Optional
from functools import partial

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S, CoefLike

from ..mesh import HomogeneousMesh
from ..functionspace.space import FunctionSpace as _FS
from ..utils import process_coef_func, is_scalar, is_tensor, fill_axis
from ..functional import bilinear_integral, linear_integral, get_semilinear_coef
from .integrator import (
    NonlinearInt, OpInt, CellInt,
    enable_cache
)


class ScalarNonlinearMassAndDiffusionIntegrator(NonlinearInt, OpInt, CellInt):
    """
    The ScalarNonlinearMassAndDiffusionIntegrator class implements a nonlinear system,
    but now can only handle the case
    - \nabla^2 \varphi - \rho = f_1
    - \nabla \rho \cdot \nabla \varphi - \rho^2 = f_2
    where \varphi is the primary variable and \rho is the secondary variable.

    Parameters
        coef1: CoefLike, optional
            The coefficient function for the first equation, which is a function of the primary variable \varphi.
        coef2: CoefLike, optional
            The coefficient function for the second equation, which is a function of the secondary variable \rho.
        grad_var: int, default=0
            The variable to differentiate with respect to, 0 for \varphi and 1 for \rho.
        q: int, optional
            The order of the quadrature formula to use for numerical integration.
        index: Index, default=_S
            The index of the mesh entity to integrate over, default is the space index.

    """
    def __init__(self,
                 coef1: Optional[CoefLike]=None,
                 coef2: Optional[CoefLike]=None,
                 grad_var: int = 0,
                 q: Optional[int]=None, *,
                 index: Index=_S,
                 batched: bool=False) -> None:
        super().__init__()
        self.grad_kernel_func = None
        self.grad_var = grad_var
        self.coef1 = coef1
        self.phi_h = coef1.phi_h if coef1 is not None else None
        self.coef2 = coef2
        self.rho_h = coef2.rho_h if coef2 is not None else None

        self.q = q
        self.index = index
        self.batched = batched

    @enable_cache
    def to_global_dof(self, space: _FS) -> TensorLike:
        return space.cell_to_dof()[self.index]

    @enable_cache
    def fetch(self, space: _FS):
        """
        Fetch the necessary data for the integrator from the function space.

        Parameters
            space: _FS
                The function space from which to fetch the data.

        Returns
            bcs: TensorLike
                The quadrature points in the mesh.
            ws: TensorLike
                The weights associated with the quadrature points.
            phi: TensorLike
                The basis functions evaluated at the quadrature points.
            gphi: TensorLike
                The gradients of the basis functions evaluated at the quadrature points.
            cm: TensorLike
                The cell measure for the integration.
            index: Index
                The index of the mesh entity to integrate over.
        """
        q = self.q
        index = self.index
        mesh = getattr(space, 'mesh', None)

        if not isinstance(mesh, HomogeneousMesh):
            raise RuntimeError("The ScalarMassIntegrator only support spaces on"
                               f"homogeneous meshes, but {type(mesh).__name__} is"
                               "not a subclass of HomoMesh.")

        cm = mesh.entity_measure('cell', index=index)
        qf = mesh.quadrature_formula(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = space.basis(bcs, index=index)
        gphi = space.grad_basis(bcs, index=index, variable='x')
        return bcs, ws, phi, gphi, cm, index

    def assembly(self, space: _FS) -> TensorLike:
        """
        Assemble the integrator for the given function space.
        Parameters
            space: _FS
                The function space for which to assemble the integrator.

        Returns
            A: TensorLike
                The assembled matrix representing the bilinear form.
            F: TensorLike
                The assembled vector representing the linear form.

        """
        mesh = getattr(space, 'mesh', None)
        bcs, ws, phi, _, cm, index = self.fetch(space)

        phi_h = self.phi_h
        coef1 = self.coef1
        coef1 = process_coef_func(coef1, bcs=bcs, mesh=mesh, etype='cell', index=index)
        rho_h = self.rho_h
        coef2 = self.coef2
        coef2 = process_coef_func(coef2, bcs=bcs, mesh=mesh, etype='cell', index=index)

        if self.grad_kernel_func is not None:
            pass
        else:
            # TODO: 是否要除次数
            phi_h_ = self.phi_h[space.cell_to_dof()]
            rho_h_ = self.rho_h[space.cell_to_dof()]
            A, F = self.auto_grad(space, phi_h_, coef1, rho_h_, coef2, batched=self.batched)

        return A, F


    def cell_integral_phi(self, varphi, rho, gphi, cm,
                          coef1, coef2, phi,ws, batched) -> TensorLike:
        # TODO: 补充更一般的形式
        val_t1 = bm.einsum('l,qld->qd', varphi, gphi)
        val_t2 = bm.einsum('l,qld->qd', rho, gphi)
        val1 = bm.einsum('qd, qd->q', val_t1, val_t2)
        val11 = bm.einsum('q, ql, q -> l', ws, phi[0], val1) * cm * coef1
        val2 = (bm.einsum('i, qi -> q', rho, phi[0]))**2
        val3 = bm.einsum('q, qi, q -> i', ws, phi[0], val2) * cm * coef2
        return val3+val11

    def cell_integral_rho(self, rho, varphi, gphi, cm,
                          coef1, coef2, phi, ws, batched) -> TensorLike:
        # TODO: 补充更一般的形式
        val_t1 = bm.einsum('l,qld->qd', varphi, gphi)
        val_t2 = bm.einsum('l,qld->qd', rho, gphi)
        val1 = bm.einsum('qd, qd->q', val_t1, val_t2)
        val11 = bm.einsum('q, ql, q -> l', ws, phi[0], val1) * cm * coef1
        val2 = (bm.einsum('i, qi -> q', rho, phi[0])) ** 2
        val3 = bm.einsum('q, qi, q -> i', ws, phi[0], val2) * cm * coef2
        return val3 + val11

    def auto_grad(self, space, phi_h_, coef1, rho_h_, coef2, batched) -> TensorLike:
        """
        Automatically compute the gradient of the integrator based on the specified variable.
        Parameters
            space: _FS
                The function space for which to compute the gradient.
            phi_h_: TensorLike
                The primary variable \varphi evaluated at the quadrature points.
            coef1: CoefLike
                The coefficient function for the first equation, which is a function of \varphi.
            rho_h_: TensorLike
                The secondary variable \rho evaluated at the quadrature points.
            coef2: CoefLike
                The coefficient function for the second equation, which is a function of \rho.
        batched: bool
            Whether to use batched operations for the computation.
        Returns
            A: TensorLike
                The assembled matrix representing the bilinear form.
            F: TensorLike
                The assembled vector representing the linear form.

        """
        bcs, ws, phi, gphi, cm, index = self.fetch(space)
        grad_var = self.grad_var

        if grad_var == 0:
            cell_integral = partial(self.cell_integral_phi,
                                    coef1=coef1, coef2=coef2, phi=phi, ws=ws, batched=batched)
        elif grad_var == 1:
            cell_integral = partial(self.cell_integral_rho,
                                    coef1=coef1, coef2=coef2, phi=phi, ws=ws, batched=batched)

        fn_A = bm.vmap(bm.jacfwd(cell_integral))
        fn_F = bm.vmap(cell_integral)

        if grad_var == 0:
            return fn_A(phi_h_, rho_h_, gphi ,cm), -fn_F(phi_h_, rho_h_, gphi ,cm)
        elif grad_var == 1:
            return fn_A(rho_h_, phi_h_, gphi ,cm), -fn_F(rho_h_, phi_h_, gphi ,cm)
